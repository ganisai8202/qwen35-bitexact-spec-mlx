"""Where does a k=1 MTP speculative cycle spend its time? Phase timings with a sync after each phase:
draft (MTP catch-up + trimmed head), verify (target T=2 forward with DeltaNet capture + full head), host (accept +
rewind + bookkeeping). Compared with a plain decode step. Uses the same setup as specmtp.py (fp16, qmvt kernel)."""
import os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
import qmvt, specmlx, specmtp
from specmlx import rewind
from bench import PROMPTS, chat

qmvt.install()
target, tok = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
target.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, target.parameters()))
lm = target.language_model; body, full_head, embed = lm.model, lm.lm_head, lm.model.embed_tokens
mtp = specmtp.load_mtp(lm.args, mx.float16)
head = specmtp.Head(lm.lm_head, specmtp.frequent_vocab(tok, 65536))
ids = tok.encode(chat(PROMPTS["math_alg"]), add_special_tokens=False)
tc, mc = make_prompt_cache(target), KVCache()
hp = body(mx.array([ids[:-1]]), cache=tc); hs = [hp[:, i:i + 1] for i in range(hp.shape[1])]
mtp(mx.concatenate(hs[:-1], axis=1), embed(mx.array([ids[1:-1]])), mc); mx.eval([c.state for c in tc] + [mc.state])
toks, m = list(ids), len(ids) - 2
T = {"draft": [], "verify": [], "host": [], "cycle": []}
for it in range(60):
    t0 = time.perf_counter()
    c = len(toks) - 1; q = list(range(m + 1, c + 1))
    o = mtp(mx.concatenate([hs[p - 1] for p in q], axis=1), embed(mx.array([[toks[p] for p in q]])), mc)
    d = head.argmax(o); mx.eval(d); t1 = time.perf_counter()
    specmlx.CAPTURE = True
    hv = body(mx.concatenate([mx.array([toks[-1]]), d.reshape(1)])[None], cache=tc)
    specmlx.CAPTURE = False
    g = mx.argmax(full_head(hv)[0], axis=-1); mx.eval(g, hv); t2 = time.perf_counter()
    dl, gl = [d.item()], g.tolist(); i = 1 if dl[0] == gl[0] else 0
    new = dl[:i] + [gl[i]]; toks += new; hs += [hv[:, j:j + 1] for j in range(i + 1)]
    rewind(tc, 1 - i, snap_idx=i); m = c
    t3 = time.perf_counter()
    if it >= 10:
        T["draft"].append((t1 - t0) * 1e3); T["verify"].append((t2 - t1) * 1e3); T["host"].append((t3 - t2) * 1e3)
        T["cycle"].append((t3 - t0) * 1e3)
med = {k: round(statistics.median(v), 2) for k, v in T.items()}
# plain step for reference (synced)
cache = make_prompt_cache(target); target(mx.array([ids]), cache=cache); mx.eval([c.state for c in cache])
y, ps = mx.array([[ids[-1]]]), []
for _ in range(40):
    t0 = time.perf_counter(); y = mx.argmax(target(y, cache=cache)[:, -1], axis=-1).reshape(1, 1); mx.eval(y); ps.append((time.perf_counter() - t0) * 1e3)
print("k=1 cycle phases (ms, synced):", med, "| plain synced step:", round(statistics.median(ps[5:]), 2))
