"""Paper table: multi-token verification kernels on M1 Pro, Qwen3.5-9B shapes, T=1..4, cost relative to MLX T=1.
Compared: MLX quantized_matmul (what MLX dispatches here: T separate qmv passes), MLX's own qmv_wide (ported, best
k_lanes), and qmvt (ours; exact at T=2,3). Each cell is the median of 7 runs of 40 independent calls per eval.
Also checks bit-exactness against MLX's single-token kernel. Prints CSV to results/kernel_compare.csv."""
import csv, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from qmvt import qmv_t
from qmvwide import qmv_wide

REPS = 40
m, _ = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
m.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, m.parameters()))
L = m.language_model.model.layers
mats = {"deltanet.in_proj_qkv 4096x8192": L[0].linear_attn.in_proj_qkv, "mlp.up 4096x12288": L[3].mlp.up_proj,
        "mlp.down 12288x4096": L[3].mlp.down_proj, "lm_head 4096x248320": m.language_model.lm_head}


def per_call(fn, xs, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval([fn(x) for x in xs]); ts.append((time.perf_counter() - t0) * 1e3 / REPS)
    return statistics.median(ts)


rows = []
for name, Lin in mats.items():
    W, S, B = Lin["weight"], Lin["scales"], Lin["biases"]; K = W.shape[1] * 8
    mlx_mm = lambda x: mx.quantized_matmul(x, W, S, B, transpose=True, group_size=64, bits=4)
    x1 = [mx.random.normal((1, K)).astype(mx.float16) for _ in range(REPS)]; mx.eval(x1)
    t1 = per_call(mlx_mm, x1)
    for T in (2, 3, 4):
        xs = [mx.random.normal((T, K)).astype(mx.float16) for _ in range(REPS)]; mx.eval(xs)
        ref_rows = mx.concatenate([mlx_mm(xs[0][t:t + 1]) for t in range(T)]).astype(mx.float32)
        exact = lambda y: bool(mx.all(y.astype(mx.float32) == ref_rows).item())
        wide = {kl: per_call(lambda x: qmv_wide(x, W, S, B, kl), xs) for kl in (4, 8, 16, 32)}
        kl_best = min(wide, key=wide.get)
        r = {"matrix": name, "T": T, "mlx_T1_ms": round(t1, 4),
             "mlx_dispatch_x": round(per_call(mlx_mm, xs) / t1, 3),
             "mlx_qmv_wide_x": round(wide[kl_best] / t1, 3), "qmv_wide_k_lanes": kl_best,
             "qmv_wide_exact": exact(qmv_wide(xs[0], W, S, B, kl_best)),
             "qmvt_x": round(per_call(lambda x: qmv_t(x, W, S, B), xs) / t1, 3),
             "qmvt_exact": exact(qmv_t(xs[0], W, S, B))}
        rows.append(r)
        print(r, flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results/kernel_compare.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/kernel_compare.csv")
