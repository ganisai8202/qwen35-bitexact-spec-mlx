"""First-principles experiments on the M1 Pro (run with the GPU otherwise idle).

1) WIRED: does pinning the model in memory (mx.set_wired_limit, as stock mlx_lm.generate does) change decode speed?
   Alternating rounds: plain greedy and exact speculative decoding, with and without the wired limit.
2) DECOMP: where do the ~5 ms/token that are not weight streaming go? Time per token of:
     full   - the real T=1 forward (async pipelined, like decoding)
     dep    - the model's matmuls only, in layer order, each depending on the previous (minimal glue)
     dep+pf - dep, plus a concurrent read of the first MB of the NEXT matrix (warm the 24 MB system cache)
     ind    - the same matmuls with independent inputs (pure streaming, no dependency bubbles)
   full - dep = non-matmul ops (norms, gates, recurrence, conv...); dep - ind = dependency/ramp bubbles.
"""
import os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import qmvt, specfast, specmtp
from specmlx import EOS, plain
from specpipe import spec_pipe
from bench import chat, PROMPTS, thermal_state

qmvt.install(); specfast.install()
target, tok = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
target.update(tree_map(lambda q: q.astype(mx.float16) if q.dtype == mx.bfloat16 else q, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
lm = target.language_model
mtp = specmtp.load_mtp(lm.args, mx.float16)
head = specmtp.Head(specmtp.head_module(lm), specmtp.frequent_vocab(tok, 65536), specmtp.full_head_fn(lm))
prompts = [tok.encode(chat(q), add_special_tokens=False) for q in PROMPTS.values()]
spec_pipe(target, mtp, head, prompts[0], 16, 1)

# ---------------- 1) WIRED
if '--decomp-only' not in sys.argv:
  pass
wired = mx.metal.device_info()["max_recommended_working_set_size"]
DO_WIRED = "--decomp-only" not in sys.argv
print(f"thermal {thermal_state()} | recommended working set {wired / 2**30:.1f} GiB", flush=True)
res = {}
for rnd in range(2 if DO_WIRED else 0):
    for setting in (["unwired", "wired"] if rnd == 0 else ["wired", "unwired"]):
        old = mx.set_wired_limit(wired if setting == "wired" else 0)
        p = statistics.mean(plain(target, ids, 200)[1] for ids in prompts)
        s = statistics.mean(spec_pipe(target, mtp, head, ids, 200, 0, 2)[1] for ids in prompts)
        res.setdefault(setting, []).append((p, s))
        print(f"WIRED round {rnd} {setting:<8} plain {p:.2f}  exact-spec {s:.2f}", flush=True)
mx.set_wired_limit(0)
for k, v in res.items():
    print(f"WIRED mean {k:<8} plain {statistics.mean(x for x, _ in v):.2f}  exact-spec {statistics.mean(y for _, y in v):.2f}")

# ---------------- 2) DECOMP
layers = lm.model.layers
plan = []   # (kind, weights...) in execution order
for L in layers:
    if L.is_linear:
        A = L.linear_attn
        plan.append(("proj", [A.in_proj_qkv, A.in_proj_z, A.in_proj_b, A.in_proj_a], A.out_proj))
    else:
        A = L.self_attn
        plan.append(("proj", [A.q_proj, A.k_proj, A.v_proj], A.o_proj))
    plan.append(("mlp", [L.mlp.gate_proj, L.mlp.up_proj], L.mlp.down_proj))
plan.append(("head", [specmtp.head_module(lm)], None))
qm = lambda x, Lw: mx.quantized_matmul(x, Lw["weight"], Lw["scales"], Lw["biases"], transpose=True, group_size=64, bits=4)
mats = [m for _, ins, out in plan for m in ins + ([out] if out is not None else [])]


def chain(x, prefetch=False):
    dummies = []
    flat = []
    for pi, (kind, ins, out) in enumerate(plan):
        ys = [qm(x, m) for m in ins]
        if kind == "head":
            flat.append(ys[0]); break
        if kind == "proj":
            v = ys[0][..., :out["weight"].shape[1] * 8]          # value slice feeds the output projection
            x = x + qm(v, out)
        else:
            x = x + qm(ys[0] * ys[1], out)
        if prefetch:   # touch the first ~1 MB of the next layer's first matrix, concurrently
            nxt = plan[min(pi + 1, len(plan) - 1)][1][0]["weight"]
            dummies.append(nxt[: max(1, (1 << 18) // nxt.shape[1])].sum())
    return flat[0] if not dummies else flat[0] + 0 * mx.stack(dummies).sum()


def per_token(fn, reps=12):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts[2:])


x0 = mx.random.normal((1, 1, 4096)).astype(mx.float16); mx.eval(x0)
# full forward, async-pipelined decode (same as plain()) in ms/token
cache = make_prompt_cache(target); target(mx.array([prompts[0]]), cache=cache); mx.eval([c.state for c in cache])
y = mx.array([[prompts[0][-1]]]); n = 60
t0 = time.perf_counter()
nxt = mx.argmax(target(y, cache=cache)[:, -1], axis=-1)
for _ in range(n):
    mx.async_eval(nxt); cur = nxt; nxt = mx.argmax(target(cur.reshape(1, 1), cache=cache)[:, -1], axis=-1); cur.item()
full = (time.perf_counter() - t0) * 1e3 / n
dep = per_token(lambda: chain(x0))
dep_pf = per_token(lambda: chain(x0, prefetch=True))
xs = {m["weight"].shape[1] * 8: mx.random.normal((1, 1, m["weight"].shape[1] * 8)).astype(mx.float16) for m in mats}
mx.eval(list(xs.values()))
ind = per_token(lambda: [qm(xs[m["weight"].shape[1] * 8], m) for m in mats])
gb = sum(m["weight"].nbytes + m["scales"].nbytes + m["biases"].nbytes for m in mats) / 1e9
print(f"DECOMP weights/token {gb:.2f} GB | full {full:.2f} ms | dep-matmuls {dep:.2f} | dep+prefetch {dep_pf:.2f} | "
      f"independent {ind:.2f} ms ({gb / ind * 1e3:.0f} GB/s)")
print(f"DECOMP non-matmul ops ~{full - dep:.2f} ms | dependency bubbles ~{dep - ind:.2f} ms | prefetch effect {dep - dep_pf:+.2f} ms")
