"""Self-speculative decoding with Qwen3.5-9B's own MTP head, plus a trimmed-vocabulary draft head (MLX).

Draft step (llama.cpp b10969 graph_mtp / vLLM Qwen3-Next MTP):
    o = mtp.norm( Layer( fc([norm_e(embed(tok)); norm_h(h)]) ) ),   logits = head(o)
where h is the target's final *normed* hidden at the previous position (or the previous draft's o, when chaining).
Each cycle starts with a catch-up pass writing the newly committed positions (with the TARGET's hiddens) into the
MTP layer's own KV cache. That pass also yields draft #1. Chained drafts are trimmed after verification.

Trimmed head (novel bit): the draft only scores the 32k most frequent token ids (75 MB instead of the 572 MB 248k
head, which would be ~80% of an MTP draft step). The target still verifies with the full head, so output is
unchanged: a draft that needed a rare token is simply rejected.

python specmtp.py [-k 1 2 3] [--vocab 32768] [--full-head]
"""
import argparse, collections, glob, json, os, sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import KVCache, make_prompt_cache

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import specmlx  # capture-mode DeltaNet + rewind (validated lossless in specmlx.py)
from specmlx import rewind, plain, EOS
from bench import PROMPTS, chat, cooldown, thermal_state


class MTP(nn.Module):
    def __init__(self, args):
        super().__init__()
        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layer = qwen3_5.DecoderLayer(args, layer_idx=args.full_attention_interval - 1)  # full attention
        self.norm = nn.RMSNorm(H, eps=eps)

    def __call__(self, h, emb, cache):
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(h)], axis=-1))
        mask = "causal" if x.shape[1] > 1 else None
        return self.norm(self.layer(x, mask=mask, cache=cache))


def head_module(lm):
    """The quantized module holding the output head: lm_head, or embed_tokens when embeddings are tied (4B/2B)."""
    return lm.model.embed_tokens if getattr(lm.args, "tie_word_embeddings", False) else lm.lm_head


def full_head_fn(lm):
    m = head_module(lm)
    return m.as_linear if m is lm.model.embed_tokens else m


def load_mtp(args, dtype, path=None):
    raw = mx.load(path or os.path.join(HERE, "models/mtp/mtp.safetensors"))
    w = {}
    for k, v in raw.items():
        k = k[len("mtp."):].replace("layers.0.", "layer.")
        if v.ndim == 1:  # every 1-D MTP tensor is an RMSNorm weight (incl. pre_fc_norm_embedding/hidden)
            v = v.astype(mx.float32) + 1.0  # Qwen3.5 zero-centred RMSNorm (mlx-lm sanitize does the same)
        w[k] = v.astype(dtype)
    m = MTP(args)
    m.load_weights(list(w.items()), strict=True)
    nn.quantize(m, group_size=64, bits=4)  # same 4-bit affine format as the target
    mx.eval(m.parameters())
    return m


def frequent_vocab(tok, n):
    """Token ids ranked by frequency over English prose (wikitext), the model's own math answers, and Python code.
    Cached in models/vocab_{n}.json."""
    cache_f = os.path.join(HERE, f"models/vocab_{n}.json")
    if os.path.exists(cache_f):
        return mx.array(json.load(open(cache_f)))
    cnt = collections.Counter()
    texts = []
    for f in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/**/*.parquet"), recursive=True)[:2]:
        try:
            import pyarrow.parquet as pq
            texts += [t for t in pq.read_table(f).column("text").to_pylist()[:20000] if t]
        except Exception:
            pass
    for f in glob.glob(os.path.join(HERE, "../surgery/results/*/*.jsonl")):
        texts += [json.loads(l)["text"] for l in open(f)][:300]
    for f in sorted(glob.glob(os.path.join(HERE, "../surgery/.venv/lib/python3.12/site-packages/mlx_lm/**/*.py"), recursive=True))[:200]:
        texts.append(open(f).read())
    for t in texts:
        cnt.update(tok.encode(t, add_special_tokens=False))
    keep = [i for i, _ in cnt.most_common(n)]
    specials = [i for i in range(len(tok.get_vocab())) if i >= 248000] + list(EOS)  # control tokens live at the top
    ids = sorted(set(keep + [i for i in specials if i < 248320]))[:n + 512]
    total = sum(cnt.values())
    print(f"trimmed vocab: {len(ids)} ids cover {sum(cnt[i] for i in ids) / total:.4%} of {total} corpus tokens "
          f"({len(texts)} texts)", flush=True)
    json.dump(ids, open(cache_f, "w"))
    return mx.array(ids)


class Head:
    """Draft head: full lm_head, or a row-slice of its quantized weights (trimmed vocabulary)."""
    def __init__(self, lm_head, ids=None, full=None):
        self.full, self.ids = full or lm_head, ids
        if ids is not None:
            self.w, self.s, self.b = lm_head["weight"][ids], lm_head["scales"][ids], lm_head["biases"][ids]
            mx.eval(self.w, self.s, self.b)

    def argmax(self, o):
        if self.ids is None:
            return mx.argmax(self.full(o[..., -1:, :]), axis=-1).reshape(())
        lg = mx.quantized_matmul(o[..., -1:, :], self.w, self.s, self.b, transpose=True, group_size=64, bits=4)
        return self.ids[mx.argmax(lg, axis=-1).reshape(())]   # views only: x[0] is a gather in MLX


def spec_mtp(target, mtp, head, ids, n, k, kmax=3, on_tokens=None):
    """k > 0: fixed draft length. k == 0: adaptive (sync-free): after a fully accepted draft k += 1, after a
    rejection at position i, k = i + 1. The cap comes from a running first-draft acceptance (EMA): <0.8 -> 1,
    <0.9 -> 2, else kmax."""
    lm = target.language_model
    body, full_head, embed = lm.model, full_head_fn(lm), lm.model.embed_tokens
    tc, mc = make_prompt_cache(target), KVCache()
    hp = body(mx.array([ids[:-1]]), cache=tc)  # normed hiddens h_0..h_{P-2}
    hs = [hp[:, i:i + 1] for i in range(hp.shape[1])]
    # MTP prefill: positions q=1..P-2 use (h_{q-1}, x_q)
    if len(ids) > 2:
        mtp(mx.concatenate(hs[:-1], axis=1), embed(mx.array([ids[1:-1]])), mc)
    mx.eval([c.state for c in tc] + [mc.state])
    toks = list(ids)          # committed tokens; toks[-1] = y (not yet processed by the target)
    m = len(ids) - 2          # last position the MTP cache holds
    out, cycles, accepted, drafted = [], 0, 0, 0
    adaptive, kc = k == 0, (2 if k == 0 else k)
    ema = 0.85  # running first-draft acceptance; caps the draft length on hard-to-predict text
    t0 = time.perf_counter()
    while len(out) < n:
        k = kc
        c = len(toks) - 1     # position of y
        # 1) catch-up (positions m+1..c) with target hiddens -> draft 1
        q = list(range(m + 1, c + 1))
        o = mtp(mx.concatenate([hs[p - 1] for p in q], axis=1), embed(mx.array([[toks[p] for p in q]])), mc)
        d = [head.argmax(o)]
        # 2) chained drafts
        for _ in range(k - 1):
            o = mtp(o[:, -1:], embed(d[-1].reshape(1, 1)), mc)
            d.append(head.argmax(o))
        # 3) verify [y, d1..dk] in one target forward (DeltaNet capture for rollback)
        specmlx.CAPTURE = True
        vin = mx.concatenate([mx.array([toks[-1]])] + [t.reshape(1) for t in d])
        hv = body(vin[None], cache=tc)
        specmlx.CAPTURE = False
        g = mx.argmax(full_head(hv), axis=-1).reshape(-1)
        dl, gl = mx.stack(d).tolist(), g.tolist()   # the one host sync per cycle
        i = 0
        while i < k and dl[i] == gl[i]:
            i += 1
        new = dl[:i] + [gl[i]]
        out += new; toks += new; cycles += 1; accepted += i; drafted += k
        if on_tokens:
            on_tokens(new)
        if adaptive:
            ema = 0.9 * ema + 0.1 * (i >= 1)
            kcap = 1 if ema < 0.8 else (min(2, kmax) if ema < 0.9 else kmax)
            kc = min(k + 1, kcap) if i == k else max(1, min(i + 1, kcap))
        hs += [hv[:, j:j + 1] for j in range(i + 1)]   # target hiddens for positions c..c+i
        rewind(tc, k - i, snap_idx=i)
        if k > 1:
            mc.trim(k - 1)                              # drop chained draft positions
        m = c
        if EOS & set(new):
            break
    dt = time.perf_counter() - t0
    if EOS & set(out):
        out = out[:min(out.index(e) for e in EOS if e in out) + 1]
    return out[:n], len(out) / dt, accepted / drafted, len(out) / cycles


def main():
    a = argparse.ArgumentParser()
    a.add_argument("-k", type=int, nargs="+", default=[1, 2, 3]); a.add_argument("-n", type=int, default=256)
    a.add_argument("--vocab", type=int, default=32768); a.add_argument("--full-head", action="store_true")
    a.add_argument("--name", default="specmtp"); a.add_argument("--cool", type=int, default=20)
    a.add_argument("--kernel", choices=["mlx", "qmvt"], default="qmvt", help="matmul kernel for 2..8-token calls")
    a.add_argument("--kmax", type=int, default=3, help="max draft length for adaptive k (-k 0)")
    a.add_argument("--compile", action="store_true", help="mx.compile DeltaNet capture-mode layers (specfast.py)")
    a.add_argument("--fuse", action="store_true", help="fuse projections sharing an input (fuse.py)")
    a.add_argument("--pipe", action="store_true", help="optimistic prebuild of the next cycle (specpipe.py)")
    a.add_argument("--fused-gdn", action="store_true", help="one fused Metal kernel per DeltaNet middle (fusedgdn.py)")
    a.add_argument("--model", default=os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
    a.add_argument("--mtp", default=None, help="MTP safetensors (default models/mtp/mtp.safetensors = 9B)")
    a = a.parse_args()
    if a.compile:
        import specfast
        specfast.install()
    if a.kernel == "qmvt":
        import qmvt
        qmvt.install()
    target, tok = load(a.model)
    target.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, target.parameters()))
    EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
    if a.fuse:
        import fuse
        print("fused groups:", fuse.apply(target), flush=True)
    if a.fused_gdn:
        import fusedgdn
        fusedgdn.install(target)
    mtp = load_mtp(target.language_model.args, mx.float16, a.mtp)
    lm = target.language_model
    head = Head(head_module(lm), None if a.full_head else frequent_vocab(tok, a.vocab), full_head_fn(lm))
    cooldown(a.cool)
    warm = tok.encode(chat("Say hi."), add_special_tokens=False)
    plain(target, warm, 8); spec_mtp(target, mtp, head, warm, 8, 1)
    res = {}
    for name, qtext in PROMPTS.items():
        ids = tok.encode(chat(qtext), add_special_tokens=False)
        ref, tps0 = plain(target, ids, a.n)
        row = {"plain_tps": round(tps0, 2)}
        for k in a.k:
            fn = spec_mtp
            if a.pipe:
                from specpipe import spec_pipe as fn
            got, tps, acc, tpc = fn(target, mtp, head, ids, a.n, k, a.kmax)
            same = next((j for j, (x, z) in enumerate(zip(ref, got)) if x != z), min(len(ref), len(got)))
            row["auto" if k == 0 else f"k{k}"] = {"tps": round(tps, 2), "speedup": round(tps / tps0, 2), "accept": round(acc, 3),
                            "tok_per_cycle": round(tpc, 2), "identical_prefix": f"{same}/{len(ref)}"}
        res[name] = row
        print(name, json.dumps(row), flush=True)
    print("thermal after", thermal_state())
    json.dump(res, open(os.path.join(HERE, f"results/{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
