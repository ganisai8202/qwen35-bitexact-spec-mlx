"""Speculative decoding for Qwen3.5 (hybrid Gated-DeltaNet + attention) in MLX, with recurrent-state snapshots.

mlx-lm 0.32.2 refuses speculative decoding here: DeltaNet's ArraysCache is not trimmable. This loop makes it work:
  * verify: the target runs [y, d1..dk] in ONE batched forward (each weight read once for k+1 tokens). DeltaNet
    layers run in capture mode: projections stay batched and only the cheap recurrence is stepped per position,
    so the (conv_state, ssm_state) after every position is kept and rejected drafts rewind with no recompute.
  * draft: MLX arrays are immutable, so a snapshot of the draft's recurrent state is just a reference.
  * one host sync per cycle: the k draft steps and the verify form one lazy graph, then acceptance is counted.
Greedy and lossless by construction: every emitted token is the target's argmax.

python specmlx.py [--draft PATH] [-k 4] [--plain-only]
"""
import argparse, json, os, sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache, make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import PROMPTS, chat, cooldown, thermal_state  # same prompts/format as bench.py

CAPTURE = False
_orig_gdn = qwen3_5.GatedDeltaNet.__call__


def _gdn_capture(self, inputs, mask=None, cache=None):
    """GatedDeltaNet forward that also records (conv_state, ssm_state) after each of the S positions."""
    if not CAPTURE or cache is None or inputs.shape[1] == 1:
        return _orig_gdn(self, inputs, mask, cache)
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
    n_keep = self.conv_kernel_size - 1
    conv_state = cache[0] if cache[0] is not None else mx.zeros((B, n_keep, self.conv_dim), dtype=inputs.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_snaps = [conv_input[:, j + 1:j + 1 + n_keep] for j in range(S)]
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
        [self.num_k_heads, self.num_k_heads, self.num_v_heads], [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim ** -0.5, eps=1e-6)
    state, outs, states = cache[1], [], []
    for t in range(S):
        o, state = gated_delta_update(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], a[:, t:t + 1], b[:, t:t + 1],
                                      self.A_log, self.dt_bias, state, None, use_kernel=True)
        outs.append(o); states.append(state)
    cache[0], cache[1] = conv_snaps[-1], state
    cache.snaps = list(zip(conv_snaps, states))
    out = self.norm(mx.concatenate(outs, axis=1), z)
    return self.out_proj(out.reshape(B, S, -1))


qwen3_5.GatedDeltaNet.__call__ = _gdn_capture


def rewind(cache, n_drop, snap_idx=None, draft_snaps=None):
    """Drop the last n_drop positions. DeltaNet: restore a snapshot; attention KV: trim."""
    for li, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            if snap_idx is not None:
                c[0], c[1] = c.snaps[snap_idx]
            elif draft_snaps is not None:
                c[0], c[1] = draft_snaps[li]
        elif n_drop:
            c.trim(n_drop)


EOS = set()


def plain(model, ids, n):
    """Plain greedy decode (reference for speed and for exactness)."""
    cache = make_prompt_cache(model)
    model(mx.array([ids[:-1]]), cache=cache); mx.eval([c.state for c in cache])
    y, out = mx.array([ids[-1]]), []
    t0 = time.perf_counter()
    nxt = mx.argmax(model(y[None], cache=cache)[:, -1], axis=-1)
    for _ in range(n):
        mx.async_eval(nxt)                       # overlap host work with GPU, as mlx-lm does
        cur = nxt
        nxt = mx.argmax(model(cur[None], cache=cache)[:, -1], axis=-1)
        out.append(cur.item())
        if out[-1] in EOS:
            break
    return out, len(out) / (time.perf_counter() - t0)


def speculative(target, draft, ids, n, k):
    global CAPTURE
    tc, dc = make_prompt_cache(target), make_prompt_cache(draft)
    target(mx.array([ids[:-1]]), cache=tc); draft(mx.array([ids[:-1]]), cache=dc)
    mx.eval([c.state for c in tc + dc])
    y, pending, out, cycles, accepted = ids[-1], [], [], 0, 0
    t0 = time.perf_counter()
    while len(out) < n:
        # draft k tokens lazily, keeping a reference snapshot of the draft's recurrent state after each feed
        feed, d, dsnaps, dfed = pending + [y], [], [], []
        for j in range(k):
            logits = draft(mx.array([feed]) if j == 0 else d[-1].reshape(1, 1), cache=dc)
            d.append(mx.argmax(logits[0, -1], axis=-1))
            dsnaps.append({li: (c[0], c[1]) for li, c in enumerate(dc) if isinstance(c, ArraysCache)})
            dfed.append(len(feed) if j == 0 else 1)
        # verify [y, d1..dk] in one batched target forward (lazy graph includes the draft steps)
        CAPTURE = True
        vin = mx.concatenate([mx.array([y])] + [t.reshape(1) for t in d])
        g = mx.argmax(target(vin[None], cache=tc)[0], axis=-1)
        CAPTURE = False
        dl, gl = mx.stack(d).tolist(), g.tolist()  # the single host sync of the cycle
        i = 0
        while i < k and dl[i] == gl[i]:
            i += 1
        out += dl[:i] + [gl[i]]
        cycles += 1; accepted += i
        rewind(tc, k - i, snap_idx=i)  # target keeps [y, d1..di]
        if i < k:   # draft fed [pending+y, d1..d_{k-1}]; it must keep [pending+y, d1..di] = feeds 0..i
            rewind(dc, sum(dfed[i + 1:]), draft_snaps=dsnaps[i]); pending = []
        else:       # all accepted: draft never fed d_k
            pending = [dl[k - 1]]
        y = gl[i]
        if EOS & set(out):
            out = out[:min(out.index(e) for e in EOS if e in out) + 1]
            break
    dt = time.perf_counter() - t0
    return out[:n], len(out) / dt, accepted / (cycles * k), len(out) / cycles


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--target", default=os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
    a.add_argument("--draft", default=os.path.join(HERE, "models/Qwen3.5-0.8B-4bit"))
    a.add_argument("-k", type=int, nargs="+", default=[4]); a.add_argument("-n", type=int, default=256)
    a.add_argument("--plain-only", action="store_true"); a.add_argument("--cool", type=int, default=45)
    a.add_argument("--name", default="spec")
    a = a.parse_args()
    target, tok = load(a.target)
    EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
    draft = None if a.plain_only else load(a.draft)[0]
    waited = cooldown(a.cool)
    print(f"cooldown {waited}s, thermal {thermal_state()}", flush=True)
    warm = tok.encode(chat("Say hi."), add_special_tokens=False)
    plain(target, warm, 8)
    if draft is not None:
        speculative(target, draft, warm, 8, 2)
    res = {}
    for name, q in PROMPTS.items():
        ids = tok.encode(chat(q), add_special_tokens=False)
        ref, tps0 = plain(target, ids, a.n)
        row = {"plain_tps": round(tps0, 2)}
        if draft is not None:
            for k in a.k:
                got, tps, acc, tpc = speculative(target, draft, ids, a.n, k)
                same = next((j for j, (x, z) in enumerate(zip(ref, got)) if x != z), len(ref))
                row[f"k{k}"] = {"tps": round(tps, 2), "speedup": round(tps / tps0, 2), "accept": round(acc, 3),
                                "tok_per_cycle": round(tpc, 2), "identical_prefix": same}
        res[name] = row
        print(name, json.dumps(row), flush=True)
    print("thermal after", thermal_state())
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, f"results/{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
