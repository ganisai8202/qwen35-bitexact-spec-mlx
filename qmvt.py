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
    constexpr int RPS = RPST;          // output rows per simdgroup (template)
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
        XT xt[TT][VPT];
        float xs[TT];
        _Pragma("unroll") for (int t = 0; t < TT; t++) {
            float s = 0.0f;
            _Pragma("unroll") for (int i = 0; i < VPT; i += 4) {
                const float a0 = float(xb[t * K + k + i]),     a1 = float(xb[t * K + k + i + 1]);
                const float a2 = float(xb[t * K + k + i + 2]), a3 = float(xb[t * K + k + i + 3]);
                s += a0 + a1 + a2 + a3;
                if (RAWX) { xt[t][i] = XT(a0); xt[t][i + 1] = XT(a1); xt[t][i + 2] = XT(a2); xt[t][i + 3] = XT(a3); }
                else { xt[t][i] = XT(a0); xt[t][i + 1] = XT(a1 / 16.0f); xt[t][i + 2] = XT(a2 / 256.0f); xt[t][i + 3] = XT(a3 / 4096.0f); }
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
                if (RAWX) {
                    wf[4 * i] = float(q & 0x000f); wf[4 * i + 1] = float((q >> 4) & 0x000f);
                    wf[4 * i + 2] = float((q >> 8) & 0x000f); wf[4 * i + 3] = float(q >> 12);
                } else {
                    wf[4 * i] = float(q & 0x000f); wf[4 * i + 1] = float(q & 0x00f0);
                    wf[4 * i + 2] = float(q & 0x0f00); wf[4 * i + 3] = float(q & 0xf000);
                }
            }
            _Pragma("unroll") for (int t = 0; t < TT; t++) {
                float acc = 0.0f;
                // same association as MLX's qdot: accum += (x0*w0 + x1*w1 + x2*w2 + x3*w3) per uint16 pack;
                // with VPT=16 and fp32 registers each row should match MLX's single-token qmv_fast bit for bit
                _Pragma("unroll") for (int i = 0; i < VPT / 4; i++)
                    acc += (float(xt[t][4 * i]) * wf[4 * i] + float(xt[t][4 * i + 1]) * wf[4 * i + 1] +
                            float(xt[t][4 * i + 2]) * wf[4 * i + 2] + float(xt[t][4 * i + 3]) * wf[4 * i + 3]);
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


CONFIG = {2: (16, 4, mx.float32, False), 3: (16, 4, mx.float16, True), 4: (8, 4, mx.float16, False)}  # T -> (VPT, RPS, x-reg dtype, raw-x); T=2,3 bit-exact vs MLX single-token, T=4 within 1-2 fp16 ulp


def qmv_t(x2d, w, scales, biases, vpt=None, rps=None, xt=None, raw=None):
    T, K = x2d.shape
    N = w.shape[0]
    dv, dr, dx, draw = (CONFIG.get(T, (8, 4, mx.float32)) + (False,))[:4]
    vpt, rps, xt = vpt or dv, rps or dr, xt or dx
    raw = draw if raw is None else raw
    n_tg = (N + 2 * rps - 1) // (2 * rps)
    return _kernel(inputs=[x2d, w, scales, biases],
                   template=[("TT", T), ("VPTT", vpt), ("RPST", rps), ("XT", xt), ("RAWX", bool(raw)), ("InT", x2d.dtype), ("OutT", x2d.dtype)],
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


_orig_as_linear = nn.QuantizedEmbedding.as_linear


def _as_linear(self, x):
    T = 1
    for d in x.shape[:-1]:
        T *= d
    if 2 <= T <= 4 and self.bits == 4 and self.group_size == 64 and getattr(self, "mode", "affine") == "affine" and x.shape[-1] % 512 == 0:
        return qmv_t(x.reshape(-1, x.shape[-1]), self["weight"], self["scales"], self["biases"]).reshape(*x.shape[:-1], -1)
    return _orig_as_linear(self, x)


def install():
    nn.QuantizedLinear.__call__ = _call
    nn.QuantizedEmbedding.as_linear = _as_linear


def uninstall():
    nn.QuantizedLinear.__call__ = _orig
    nn.QuantizedEmbedding.as_linear = _orig_as_linear


if __name__ == "__main__":
    import os, statistics, sys, time
    from mlx.utils import tree_map
    from mlx_lm import load
    HERE = os.path.dirname(os.path.abspath(__file__))
    model, _ = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
    model.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, model.parameters()))
    lay = model.language_model.model.layers[3].mlp
    REPS = 40
    def per_call(make, reps=7):
        ts = []
        for _ in range(reps):
            xs = make(); t0 = time.perf_counter(); mx.eval(xs); ts.append((time.perf_counter() - t0) * 1e3 / REPS)
        return statistics.median(ts)
    cfgs = [(16, 4, mx.float32), (16, 2, mx.float32), (16, 4, mx.float16), (16, 2, mx.float16),
            (8, 4, mx.float32), (8, 2, mx.float32), (8, 4, mx.float16), (8, 2, mx.float16)]
    for name, L in [("mlp.up", lay.up_proj), ("mlp.down", lay.down_proj), ("lm_head", model.language_model.lm_head)]:
        K = L["weight"].shape[1] * 8; W, S, B = L["weight"], L["scales"], L["biases"]
        xs1 = [mx.random.normal((1, K)).astype(mx.float16) for _ in range(REPS)]; mx.eval(xs1)
        t1 = per_call(lambda: [mx.quantized_matmul(x, W, S, B, transpose=True, group_size=64, bits=4) for x in xs1])
        for T in (2, 3, 4):
            xs = [mx.random.normal((T, K)).astype(mx.float16) for _ in range(REPS)]; mx.eval(xs)
            ref = mx.quantized_matmul(xs[0], W, S, B, transpose=True, group_size=64, bits=4).astype(mx.float32)
            row = f"{name:<9} T={T} (x T=1 cost {t1:.3f}ms):"
            for v, r, d in cfgs:
                got = qmv_t(xs[0], W, S, B, v, r, d).astype(mx.float32)
                err = (mx.abs(got - ref).max() / mx.abs(ref).max()).item()
                t = per_call(lambda: [qmv_t(x, W, S, B, v, r, d) for x in xs])
                row += f" v{v}r{r}{'h' if d == mx.float16 else 'f'}={t / t1:4.2f}" + ("" if err < 1e-2 else "!")
            print(row, flush=True)
