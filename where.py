"""Attribute the ~8 ms/token that is not ideal weight streaming (M1 Pro, Qwen3.5-9B 4-bit, T=1).
A) per matrix shape: time per call (40 queued), GB/s, and ms/token = time x occurrences per token.
B) per non-matmul op in a GDN and an attention layer: time per call (40 queued) x occurrences per token."""
import os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from bench import chat
REPS = 40
m, tok = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
m.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, m.parameters()))
lm = m.language_model; Ls = lm.model.layers
def per_call(fn, mk, reps=7):
    ts = []
    for _ in range(reps):
        xs = [mk() for _ in range(REPS)]; mx.eval(xs)
        t0 = time.perf_counter(); mx.eval([fn(x) for x in xs]); ts.append((time.perf_counter() - t0) * 1e3 / REPS)
    return statistics.median(ts)

# ---- A) matmul shapes
shapes = {}
for L in Ls:
    mods = ([L.linear_attn.in_proj_qkv, L.linear_attn.in_proj_z, L.linear_attn.in_proj_b, L.linear_attn.in_proj_a, L.linear_attn.out_proj]
            if L.is_linear else [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj, L.self_attn.o_proj]) + [L.mlp.gate_proj, L.mlp.up_proj, L.mlp.down_proj]
    for q in mods:
        key = (q["weight"].shape[1] * 8, q["weight"].shape[0])
        shapes.setdefault(key, [q, 0])[1] += 1
shapes[(4096, lm.lm_head["weight"].shape[0])] = [lm.lm_head, 1]
tot = 0
print(f"{'K x N':>16} {'count':>5} {'us/call':>8} {'GB/s':>6} {'ms/token':>8}")
for (K, N), (q, cnt) in sorted(shapes.items(), key=lambda kv: -kv[1][1] * kv[0][0] * kv[0][1]):
    by = q["weight"].nbytes + q["scales"].nbytes + q["biases"].nbytes
    t = per_call(lambda x: mx.quantized_matmul(x, q["weight"], q["scales"], q["biases"], transpose=True, group_size=64, bits=4),
                 lambda: mx.random.normal((1, 1, K)).astype(mx.float16))
    tot += t * cnt
    print(f"{K:>7} x {N:<7} {cnt:>5} {t * 1e3:8.1f} {by / t / 1e6:6.0f} {t * cnt:8.2f}")
print(f"matmuls total {tot:.2f} ms/token")

# ---- B) non-matmul ops of one GDN layer (T=1, decode state) and one attention layer
G = Ls[0].linear_attn; A3 = Ls[3]
cache = make_prompt_cache(m); ids = tok.encode(chat("Explain photosynthesis."), add_special_tokens=False)
m(mx.array([ids]), cache=cache); mx.eval([c.state for c in cache])
conv_state, ssm_state = cache[0][0], cache[0][1]
H = 4096
ops = {
  "rmsnorm (input/post-attn)": (lambda x: Ls[0].input_layernorm(x), lambda: mx.random.normal((1, 1, H)).astype(mx.float16), 64),
  "gdn conv1d+silu": (lambda x: nn.silu(G.conv1d(mx.concatenate([conv_state, x], axis=1))), lambda: mx.random.normal((1, 1, G.conv_dim)).astype(mx.float16), 24),
  "gdn normalize_qk": (lambda x: normalize_qk(x, x, inv_scale=G.head_k_dim ** -0.5, eps=1e-6)[0], lambda: mx.random.normal((1, 1, G.num_k_heads, G.head_k_dim)).astype(mx.float16), 24),
  "gdn recurrence (gated_delta_update)": (lambda x: gated_delta_update(x[0], x[0], x[1], x[2], x[2], G.A_log, G.dt_bias, ssm_state, None)[0],
      lambda: (mx.random.normal((1, 1, G.num_k_heads, G.head_k_dim)).astype(mx.float16), mx.random.normal((1, 1, G.num_v_heads, G.head_v_dim)).astype(mx.float16),
               mx.random.normal((1, 1, G.num_v_heads)).astype(mx.float16)), 24),
  "gdn gated rmsnorm": (lambda x: G.norm(x, x), lambda: mx.random.normal((1, 1, G.num_v_heads, G.head_v_dim)).astype(mx.float16), 24),
  "swiglu": (lambda x: nn.silu(x) * x, lambda: mx.random.normal((1, 1, 12288)).astype(mx.float16), 32),
  "residual add": (lambda x: x + x, lambda: mx.random.normal((1, 1, H)).astype(mx.float16), 64),
  "attention block minus projections (T=1)": (None, None, 8),
}
tot_ops = 0
for name, (fn, mk, cnt) in ops.items():
    if fn is None:
        continue
    t = per_call(fn, mk)
    tot_ops += t * cnt
    print(f"{name:<38} {t * 1e3:7.1f} us/call x {cnt:>2} = {t * cnt:5.2f} ms/token")
print(f"listed non-matmul ops total {tot_ops:.2f} ms/token (attention SDPA/rope/q-k norms, embedding, head argmax not itemized)")
