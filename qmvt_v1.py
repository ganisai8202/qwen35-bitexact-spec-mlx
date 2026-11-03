"""Multi-token version of MLX's own qmv_fast (4-bit, group 64) for T = 2..8 tokens.

MLX's qmv_fast puts the token index in the grid (x += tid.x * in_vec_size), so T tokens = T full weight passes.
Here one pass serves all T: weights are loaded/masked once, then FMA'd into T register-resident token vectors.
Same arithmetic tricks as MLX: x is pre-divided by 1,16,256,4096 so nibbles are used with a mask (no shift),
and the group bias becomes sum(x)*b. Values per thread: 16 for T<=2, 8 for T>=3 (avoid register spills).

    import qmvt; qmvt.install()   # QuantizedLinear calls with 2..8 tokens go through this kernel
"""
import mlx.core as mx
import mlx.nn as nn

SOURCE = """
    constexpr int SG = 2;              // simdgroups per threadgroup
    constexpr int RPS = 4;             // output rows per simdgroup
    constexpr int VPT = VPTT;          // values per thread per block (16 or 8)
    constexpr int BLOCK = VPT * 32;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int GP = K / 64;
    const int out_row = threadgroup_position_in_grid.x * (SG * RPS) + sg * RPS;
    if (out_row >= N) return;
    const int last = min(RPS - 1, N - 1 - out_row);

    const device uint16_t* wsb = (const device uint16_t*)w + out_row * (K / 4) + lane * (VPT / 4);
    const device InT* xb = x + lane * VPT;
    const int g0 = lane * VPT / 64;

    float res[TT][RPS];
    _Pragma("unroll") for (int t = 0; t < TT; t++) _Pragma("unroll") for (int r = 0; r < RPS; r++) res[t][r] = 0.0f;

    for (int k = 0; k < K; k += BLOCK) {
        float xt[TT][VPT];
        float xs[TT];
        _Pragma("unroll") for (int t = 0; t < TT; t++) {
            float s = 0.0f;
            _Pragma("unroll") for (int i = 0; i < VPT; i += 4) {
                const float a0 = float(xb[t * K + k + i]),     a1 = float(xb[t * K + k + i + 1]);
                const float a2 = float(xb[t * K + k + i + 2]), a3 = float(xb[t * K + k + i + 3]);
                s += a0 + a1 + a2 + a3;
                xt[t][i] = a0; xt[t][i + 1] = a1 / 16.0f; xt[t][i + 2] = a2 / 256.0f; xt[t][i + 3] = a3 / 4096.0f;
            }
            xs[t] = s;
        }
        const int g = (k / 64) + g0;
        _Pragma("unroll") for (int r = 0; r < RPS; r++) {
            const int src = min(r, last);
            const device uint16_t* wr = wsb + src * (K / 4) + k / 4;
            const float sc = float(scales[(out_row + src) * GP + g]);
            const float bi = float(biases[(out_row + src) * GP + g]);
            float wf[VPT];
            _Pragma("unroll") for (int i = 0; i < VPT / 4; i++) {
                const uint16_t q = wr[i];
                wf[4 * i] = float(q & 0x000f); wf[4 * i + 1] = float(q & 0x00f0);
                wf[4 * i + 2] = float(q & 0x0f00); wf[4 * i + 3] = float(q & 0xf000);
            }
            _Pragma("unroll") for (int t = 0; t < TT; t++) {
                float acc = 0.0f;
                _Pragma("unroll") for (int i = 0; i < VPT; i++) acc += xt[t][i] * wf[i];
                res[t][r] += sc * acc + bi * xs[t];
            }
        }
    }
    _Pragma("unroll") for (int t = 0; t < TT; t++) {
        _Pragma("unroll") for (int r = 0; r <= last; r++) {
            const float v = simd_sum(res[t][r]);
            if (lane == 0) out[t * N + out_row + r] = static_cast<OutT>(v);
        }
    }
"""

_kernel = mx.fast.metal_kernel(name="qmv_t_4bit_g64", input_names=["x", "w", "scales", "biases"],
                               output_names=["out"], source=SOURCE)


def qmv_t(x2d, w, scales, biases, vpt=None):
    T, K = x2d.shape
    N = w.shape[0]
    vpt = vpt or (16 if T <= 2 else 8)
    n_tg = (N + 7) // 8
    return _kernel(inputs=[x2d, w, scales, biases],
                   template=[("TT", T), ("VPTT", vpt), ("InT", x2d.dtype), ("OutT", x2d.dtype)],
                   grid=(n_tg * 64, 1, 1), threadgroup=(64, 1, 1),
                   output_shapes=[(T, N)], output_dtypes=[x2d.dtype])[0]


def eligible(layer, x):
    T = 1
    for s in x.shape[:-1]:
        T *= s
    return (2 <= T <= 4 and getattr(layer, "bits", None) == 4 and getattr(layer, "group_size", None) == 64
            and getattr(layer, "mode", "affine") == "affine" and x.shape[-1] % 512 == 0 and "bias" not in layer)


_orig = nn.QuantizedLinear.__call__


def _call(self, x):
    if not eligible(self, x):
        return _orig(self, x)
    return qmv_t(x.reshape(-1, x.shape[-1]), self["weight"], self["scales"], self["biases"]).reshape(*x.shape[:-1], -1)


def install():
    nn.QuantizedLinear.__call__ = _call


def uninstall():
    nn.QuantizedLinear.__call__ = _orig


if __name__ == "__main__":
    import os, statistics, sys, time
    from mlx.utils import tree_map
    from mlx_lm import load
    HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
    import mmaqmv
    model, _ = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
    model.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, model.parameters()))
    lay = model.language_model.model.layers[3].mlp
    REPS = 40

    def per_call(make, reps=7):
        ts = []
        for _ in range(reps):
            xs = make(); t0 = time.perf_counter(); mx.eval(xs); ts.append((time.perf_counter() - t0) * 1e3 / REPS)
        return statistics.median(ts)
    for name, L in [("mlp.up 4096->12288", lay.up_proj), ("mlp.down 12288->4096", lay.down_proj),
                    ("lm_head 4096->248320", model.language_model.lm_head)]:
        K = L["weight"].shape[1] * 8
        W, S, B = L["weight"], L["scales"], L["biases"]
        mm = lambda x: mx.quantized_matmul(x, W, S, B, transpose=True, group_size=64, bits=4)
        for T in (1, 2, 3, 4, 5, 8):
            xs = [mx.random.normal((T, K)).astype(mx.float16) for _ in range(REPS)]; mx.eval(xs)
            row = f"{name:<22} T={T} mlx {per_call(lambda: [mm(x) for x in xs]):6.3f}"
            if T > 1:
                ref = mm(xs[0]).astype(mx.float32)
                for label, fn in [("qmvt16", lambda x: qmv_t(x, W, S, B, 16)), ("qmvt8", lambda x: qmv_t(x, W, S, B, 8)),
                                  ("mma", lambda x: mmaqmv.qmm_small(x, W, S, B))]:
                    err = (mx.abs(fn(xs[0]).astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()
                    row += f" | {label} {per_call(lambda: [fn(x) for x in xs]):6.3f}" + ("" if err < 1e-2 else f"(ERR {err:.1e})")
            print(row, flush=True)
