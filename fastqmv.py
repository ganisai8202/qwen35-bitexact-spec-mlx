"""Small-batch 4-bit quantized mat-vec for MLX (T = 2..8 tokens), as a runtime-compiled Metal kernel.

MLX 0.32.4's quantized_matmul costs ~0.7x a full weight read per extra token at T=2..8 on M1 Pro, which kills
speculative decoding. This kernel reads every packed weight once and accumulates all T dot products in registers.
For MLX affine 4-bit, group 64: y[t,n] = sum_g s[n,g] * (x[t,g] . q[n,g]) + b[n,g] * sum(x[t,g]).

    import fastqmv; fastqmv.install()   # routes QuantizedLinear calls with 2..8 tokens through the kernel
    python fastqmv.py                   # self-check against mx.quantized_matmul + timing on a real layer
"""
import mlx.core as mx
import mlx.nn as nn

PACKS = 2                      # uint32 packs per lane per iteration -> 16 weights
BLOCK = 32 * 8 * PACKS         # K values consumed per simdgroup iteration (512)
ROWS, SGS = 4, 2               # output rows per simdgroup, simdgroups per threadgroup

SOURCE = """
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int KP = K / 8, GP = K / 64;
    const int row0 = (threadgroup_position_in_grid.x * SGS + sg) * ROWS;
    float acc[TT][ROWS];
    for (int t = 0; t < TT; t++) for (int r = 0; r < ROWS; r++) acc[t][r] = 0.0f;
    for (int k0 = lane * 8 * PACKS; k0 < K; k0 += 32 * 8 * PACKS) {
        float xv[TT][8 * PACKS];
        float xs[TT];
        for (int t = 0; t < TT; t++) {
            float s = 0.0f;
            for (int i = 0; i < 8 * PACKS; i++) { xv[t][i] = float(x[t * K + k0 + i]); s += xv[t][i]; }
            xs[t] = s;
        }
        const int g = k0 / 64;
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= N) break;
            const float sc = float(scales[row * GP + g]);
            const float bi = float(biases[row * GP + g]);
            float q[8 * PACKS];
            for (int p = 0; p < PACKS; p++) {
                const uint32_t pk = w[row * KP + k0 / 8 + p];
                for (int i = 0; i < 8; i++) q[p * 8 + i] = float((pk >> (4 * i)) & 0xF);
            }
            for (int t = 0; t < TT; t++) {
                float d = 0.0f;
                for (int i = 0; i < 8 * PACKS; i++) d += xv[t][i] * q[i];
                acc[t][r] += sc * d + bi * xs[t];
            }
        }
    }
    for (int t = 0; t < TT; t++) {
        for (int r = 0; r < ROWS; r++) {
            const float v = simd_sum(acc[t][r]);
            const int row = row0 + r;
            if (lane == 0 && row < N) out[t * N + row] = static_cast<OutT>(v);
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="qmv_multi_4bit_g64", input_names=["x", "w", "scales", "biases"], output_names=["out"],
    header=f"#define SGS {SGS}\n", source=SOURCE)
CONFIG = {2: (2, 4), 3: (1, 4), 4: (1, 4), 5: (1, 2), 6: (1, 2), 7: (1, 2), 8: (1, 2)}  # T -> (PACKS, ROWS)


def qmv_multi(x2d, w, scales, biases, cfg=None):
    """x2d [T, K] (T<=8), w [N, K/8] uint32, scales/biases [N, K/64] -> [T, N]."""
    T, K = x2d.shape
    N = w.shape[0]
    packs, rows = cfg or CONFIG[T]
    n_tg = (N + rows * SGS - 1) // (rows * SGS)
    return _kernel(inputs=[x2d, w, scales, biases], template=[("TT", T), ("OutT", x2d.dtype), ("PACKS", packs), ("ROWS", rows)], grid=(n_tg * 32 * SGS, 1, 1),
                   threadgroup=(32 * SGS, 1, 1), output_shapes=[(T, N)], output_dtypes=[x2d.dtype])[0]


def eligible(layer, x):
    T = 1
    for s in x.shape[:-1]:
        T *= s
    return (2 <= T <= 8 and getattr(layer, "bits", None) == 4 and getattr(layer, "group_size", None) == 64
            and getattr(layer, "mode", "affine") == "affine" and x.shape[-1] % 512 == 0 and "bias" not in layer)


_orig = nn.QuantizedLinear.__call__


def _fast_call(self, x):
    if not eligible(self, x):
        return _orig(self, x)
    y = qmv_multi(x.reshape(-1, x.shape[-1]), self["weight"], self["scales"], self["biases"])
    return y.reshape(*x.shape[:-1], -1)


def install():
    nn.QuantizedLinear.__call__ = _fast_call


def uninstall():
    nn.QuantizedLinear.__call__ = _orig


if __name__ == "__main__":
    import os, statistics, time
    from mlx_lm import load
    HERE = os.path.dirname(os.path.abspath(__file__))
    model, _ = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
    lay = model.language_model.model.layers[3].mlp
    REPS = 40  # calls queued per eval, so launch/sync overhead amortizes away

    def per_call_ms(make, reps=7):
        ts = []
        for _ in range(reps):
            xs = make()
            t0 = time.perf_counter(); mx.eval(xs); ts.append((time.perf_counter() - t0) * 1e3 / REPS)
        return statistics.median(ts)
    for name, L in [("mlp.up 4096->12288", lay.up_proj), ("mlp.down 12288->4096", lay.down_proj),
                    ("lm_head 4096->248320", model.language_model.lm_head)]:
        K = L["weight"].shape[1] * 8
        nbytes = L["weight"].nbytes + L["scales"].nbytes + L["biases"].nbytes
        for T in (1, 2, 3, 4, 5, 8):
            xs = [mx.random.normal((T, K)).astype(L["scales"].dtype) for _ in range(REPS)]; mx.eval(xs)
            mm = lambda x: mx.quantized_matmul(x, L["weight"], L["scales"], L["biases"], transpose=True, group_size=64, bits=4)
            t_ref = per_call_ms(lambda: [mm(x) for x in xs])
            row = f"{name:<22} T={T} mlx {t_ref:6.3f} ms ({nbytes / t_ref / 1e6:5.0f} GB/s) |"
            if T > 1:
                ref = mm(xs[0])
                for cfg in [(2, 4), (1, 4), (1, 2)]:
                    got = qmv_multi(xs[0], L["weight"], L["scales"], L["biases"], cfg)
                    err = (mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max()).item()
                    t = per_call_ms(lambda: [qmv_multi(x, L["weight"], L["scales"], L["biases"], cfg) for x in xs])
                    row += f" p{cfg[0]}r{cfg[1]} {t:6.3f}" + ("" if err < 1e-2 else "(ERR)")
            print(row, flush=True)
