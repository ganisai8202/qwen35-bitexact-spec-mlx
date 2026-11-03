"""Small-batch (T<=8) 4-bit quantized matmul for MLX on Apple GPU using simdgroup 8x8 matrix units.

Why: MLX 0.32.4 on M1 Pro re-reads the weights once per token for T=2..6 (cost ~T x) and switches to a tiled
kernel costing ~8.6x a single token at T>=7, so speculative verification is never cheap. Here every weight is
read and dequantized ONCE into threadgroup memory; the T (zero-padded to 8) token rows then hit it with 8x8 MMAs.
So the cost is ~flat in T<=8 (dequant + MMA are both cheap vs the weight stream).

Layout: MLX affine 4-bit, group 64: w [N, K/8] uint32 (low nibble first), scales/biases [N, K/64].
Per threadgroup: NSG simdgroups x 32 output rows. Per K-step of 64 (= one quant group):
  Xs[8][64]          <- x rows (zero-padded), shared by the threadgroup
  Ws[sg][32][WSTR]   <- lane l dequantizes row n0+l (8 packs = 64 weights); padded stride avoids bank conflicts
  C[j] += A(Xs[:, kk:kk+8]) x B(Ws[j*8:j*8+8, kk:kk+8]^T)   for kk in 0..56 step 8, j in 0..3
"""
import mlx.core as mx
import mlx.nn as nn

NSG, WSTR, J = 4, 0, 4   # 4 simdgroups x J=4 n-tiles x 8 rows = 128 rows per threadgroup; no threadgroup memory

SOURCE = """
    // v4: register-direct fragments, no threadgroup memory, no barriers.
    // Y^T[n, t] = sum_k W[n, k] * X^T[k, t]:  A = W tile (8n x 8k), B = X^T tile (8k x 8t), C = 8n x 8t (float).
    // Fragment layout (MLX steel BaseMMAFrag): lane holds (row fm, cols fn, fn+1).
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int KP = K / 8, GP = K / 64;
    const int n_sg = (threadgroup_position_in_grid.x * NSG + sg) * (J * 8);

    simdgroup_float8x8 C[J];
    for (int j = 0; j < J; j++) C[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    int rows[J];
    for (int j = 0; j < J; j++) { const int r = n_sg + j * 8 + fm; rows[j] = r < N ? r : N - 1; }
    // x rows this lane feeds into B (cols fn, fn+1 = token index); rows >= TT contribute zeros
    const bool t0ok = fn < TT, t1ok = fn + 1 < TT;
    const device InT* x0 = x + (t0ok ? fn : 0) * K;
    const device InT* x1 = x + (t1ok ? fn + 1 : 0) * K;

    for (int k0 = 0; k0 < K; k0 += 64) {
        const int g = k0 / 64;
        uint4 wv[J][2];
        float sc[J], bi[J];
        for (int j = 0; j < J; j++) {
            const device uint4* wp4 = (const device uint4*)(w + rows[j] * KP + k0 / 8);
            wv[j][0] = wp4[0]; wv[j][1] = wp4[1];
            sc[j] = float(scales[rows[j] * GP + g]); bi[j] = float(biases[rows[j] * GP + g]);
        }
        for (int p = 0; p < 8; p++) {           // 8 k-tiles of 8 within this quant group
            const int kk = k0 + p * 8;
            simdgroup_half8x8 B;
            // B = X^T tile: lane holds rows k = fm, cols t = fn, fn+1
            thread auto& be = B.thread_elements();
            be[0] = t0ok ? half(float(x0[kk + fm])) : half(0.0f);
            be[1] = t1ok ? half(float(x1[kk + fm])) : half(0.0f);
            for (int j = 0; j < J; j++) {
                const uint pk = wv[j][p / 4][p % 4];
                simdgroup_half8x8 A;
                thread auto& ae = A.thread_elements();
                ae[0] = half(sc[j] * float((pk >> (4 * fn)) & 0xFu) + bi[j]);
                ae[1] = half(sc[j] * float((pk >> (4 * (fn + 1))) & 0xFu) + bi[j]);
                simdgroup_multiply_accumulate(C[j], A, B, C[j]);
            }
        }
    }
    for (int j = 0; j < J; j++) {
        const int n = n_sg + j * 8 + fm;
        thread auto& ce = C[j].thread_elements();
        if (n < N) {
            if (t0ok) out[fn * N + n] = static_cast<OutT>(ce[0]);
            if (t1ok) out[(fn + 1) * N + n] = static_cast<OutT>(ce[1]);
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="qmm_small_t_4bit_g64", input_names=["x", "w", "scales", "biases"], output_names=["out"],
    header=f"#include <metal_simdgroup_matrix>\n#define NSG {NSG}\n#define J {J}\n", source=SOURCE)


def qmm_small(x2d, w, scales, biases):
    """x2d [T, K] (1<=T<=8), w [N, K/8] uint32, scales/biases [N, K/64] -> [T, N]."""
    T, K = x2d.shape
    N = w.shape[0]
    n_tg = (N + 8 * J * NSG - 1) // (8 * J * NSG)
    return _kernel(inputs=[x2d, w, scales, biases], template=[("TT", T), ("InT", x2d.dtype), ("OutT", x2d.dtype)],
                   grid=(n_tg * 32 * NSG, 1, 1), threadgroup=(32 * NSG, 1, 1),
                   output_shapes=[(T, N)], output_dtypes=[x2d.dtype])[0]


def eligible(layer, x, lo=2, hi=8):
    T = 1
    for s in x.shape[:-1]:
        T *= s
    return (lo <= T <= hi and getattr(layer, "bits", None) == 4 and getattr(layer, "group_size", None) == 64
            and getattr(layer, "mode", "affine") == "affine" and x.shape[-1] % 64 == 0 and "bias" not in layer)


_orig = nn.QuantizedLinear.__call__


def _call(self, x):
    if not eligible(self, x):
        return _orig(self, x)
    return qmm_small(x.reshape(-1, x.shape[-1]), self["weight"], self["scales"], self["biases"]).reshape(*x.shape[:-1], -1)


def install():
    nn.QuantizedLinear.__call__ = _call


def uninstall():
    nn.QuantizedLinear.__call__ = _orig


if __name__ == "__main__":
    import os, statistics, time
    from mlx_lm import load
    HERE = os.path.dirname(os.path.abspath(__file__))
    model, _ = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
    lay = model.language_model.model.layers[3].mlp
    REPS = 40

    def per_call_ms(make, reps=7):
        ts = []
        for _ in range(reps):
            xs = make()
            t0 = time.perf_counter(); mx.eval(xs); ts.append((time.perf_counter() - t0) * 1e3 / REPS)
        return statistics.median(ts)
    for name, L in [("mlp.up 4096->12288", lay.up_proj), ("mlp.down 12288->4096", lay.down_proj),
                    ("lm_head 4096->248320", model.language_model.lm_head)]:
        K = L["weight"].shape[1] * 8
        mm = lambda x: mx.quantized_matmul(x, L["weight"], L["scales"], L["biases"], transpose=True, group_size=64, bits=4)
        for T in (1, 2, 4, 5, 8):
            xs = [mx.random.normal((T, K)).astype(L["scales"].dtype) for _ in range(REPS)]; mx.eval(xs)
            ref, got = mm(xs[0]), qmm_small(xs[0], L["weight"], L["scales"], L["biases"])
            err = (mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max()).item()
            t_ref = per_call_ms(lambda: [mm(x) for x in xs])
            t_new = per_call_ms(lambda: [qmm_small(x, L["weight"], L["scales"], L["biases"]) for x in xs])
            print(f"{name:<22} T={T}  mlx {t_ref:6.3f} ms   mma {t_new:6.3f} ms   rel.err {err:.1e}", flush=True)
