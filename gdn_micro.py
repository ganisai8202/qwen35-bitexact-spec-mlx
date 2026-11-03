"""Micro-benchmark of the Gated-DeltaNet middle (conv .. gated norm) on one 9B-shaped layer with random weights:
stock MLX op chain vs fusedgdn kernel. 24 calls chained through the recurrent state (like 24 layers), us/call."""
import os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
import fusedgdn
from mlx.utils import tree_map

args = qwen3_5.TextModelArgs(model_type="qwen3_5", hidden_size=4096, intermediate_size=12288, num_hidden_layers=32,
                             num_attention_heads=16, num_key_value_heads=4, rms_norm_eps=1e-6, vocab_size=1000,
                             linear_num_value_heads=32, linear_num_key_heads=16, linear_key_head_dim=128,
                             linear_value_head_dim=128, linear_conv_kernel_dim=4)
G = qwen3_5.GatedDeltaNet(args)
G.update(tree_map(lambda p: p.astype(mx.float16), G.parameters()))
G.A_log = G.A_log.astype(mx.float32)
mx.eval(G.parameters())
Hv, Dv, CD = G.num_v_heads, G.head_v_dim, G.conv_dim


def stock(qkv, z, a, b, conv, state):   # mlx_lm GatedDeltaNet.__call__ between the projections and out_proj, T=1
    B, S = 1, 1
    conv_input = mx.concatenate([conv, qkv], axis=1)
    new_conv = mx.contiguous(conv_input[:, -3:, :])
    conv_out = nn.silu(G.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(mx.split(conv_out, [G.key_dim, 2 * G.key_dim], -1),
               [G.num_k_heads, G.num_k_heads, G.num_v_heads], [G.head_k_dim, G.head_k_dim, G.head_v_dim])]
    q, k = normalize_qk(q, k, inv_scale=G.head_k_dim ** -0.5, eps=1e-6)
    out, state = gated_delta_update(q, k, v, a, b, G.A_log, G.dt_bias, state, None)
    return G.norm(out, z.reshape(B, S, Hv, Dv)), new_conv, state


def fused(qkv, z, a, b, conv, state):
    y, ss, cs = fusedgdn.fused(G, qkv.reshape(1, -1), z.reshape(1, -1), a.reshape(1, -1), b.reshape(1, -1), conv, state)
    return y, cs, ss


def bench(fn, layers=24, reps=15):
    qkv = mx.random.normal((1, 1, CD)).astype(mx.float16); z = mx.random.normal((1, 1, Hv * Dv)).astype(mx.float16)
    a = mx.random.normal((1, 1, Hv)).astype(mx.float16); b = mx.random.normal((1, 1, Hv)).astype(mx.float16)
    conv = mx.random.normal((1, 3, CD)).astype(mx.float16); st = mx.random.normal((1, Hv, Dv, 128)) * 0.1
    mx.eval(qkv, z, a, b, conv, st)
    ts, bs = [], []
    for _ in range(reps):
        s, c = st, conv
        t0 = time.perf_counter()
        for _ in range(layers):
            y, c, s = fn(qkv, z, a, b, c, s)
        t1 = time.perf_counter()
        mx.eval(y, c, s)
        ts.append((time.perf_counter() - t0) * 1e6 / layers); bs.append((t1 - t0) * 1e6 / layers)
    return statistics.median(ts[3:]), statistics.median(bs[3:])


if __name__ == "__main__":
    for name, fn in [("stock", stock), ("fused", fused), ("stock", stock), ("fused", fused)]:
        t, b = bench(fn)
        print(f"GDN middle {name:<6} {t:7.1f} us/layer total, {b:6.1f} us/layer of it building the graph (CPU)", flush=True)
