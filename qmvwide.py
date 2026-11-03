"""Port of MLX 0.32.4's own `qmv_wide_impl` (mlx/backend/metal/kernels/quantized.h; MIT license) for 4-bit
affine, group 64, so it can be benchmarked against qmvt.py on this M1 Pro. MLX ships this kernel, but on this
GPU its dispatcher keeps using T separate qmv passes for M=2..6, so it is called directly here.

Algorithm, unchanged: each output row is reduced by k_lanes lanes over K; a group is dequantized in 8-value
sub-chunks (scale*q + bias) and reused across the vecs_per_tg (=T) input vectors; shuffle-ladder reduction.
Sizes are passed by value (the original takes `constant int&`, which a JIT custom kernel cannot bind).
"""
import mlx.core as mx

SOURCE = """
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 32 / KL;
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int M = x_shape[0];
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const short k_lane = simd_lid % KL;
    const short sg_row = simd_lid / KL;
    const int out_row = threadgroup_position_in_grid.x * (results_per_simdgroup * num_simdgroups) +
                        results_per_simdgroup * simd_gid + sg_row;
    const int row = min(out_row, N - 1);
    const int in_vec_size_w = K / 2;        // bytes per 4-bit row
    const int in_vec_size_g = K / 64;
    const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
    const device InT* srow = scales + row * in_vec_size_g;
    const device InT* brow = biases + row * in_vec_size_g;
    float result[VT];
    for (int v = 0; v < VT; v++) result[v] = 0.0f;
    for (int g = k_lane; g < in_vec_size_g; g += KL) {
        const float scale = float(srow[g]);
        const float bias = float(brow[g]);
        #pragma unroll
        for (int sc = 0; sc < 64 / 8; sc++) {
            const int k0 = g * 64 + sc * 8;
            const device uint8_t* wc = wrow + k0 / 2;
            float w_dq[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                w_dq[2 * i] = scale * float(wc[i] & 0x0f) + bias;
                w_dq[2 * i + 1] = scale * float((wc[i] >> 4) & 0x0f) + bias;
            }
            #pragma unroll
            for (int v = 0; v < VT; v++) {
                const device InT* xc = x + min(v, M - 1) * K + k0;
                float acc = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) acc += float(xc[i]) * w_dq[i];
                result[v] += acc;
            }
        }
    }
    for (int v = 0; v < VT; v++) {
        if (KL >= 32) result[v] += simd_shuffle_down(result[v], 16);
        if (KL >= 16) result[v] += simd_shuffle_down(result[v], 8);
        if (KL >= 8)  result[v] += simd_shuffle_down(result[v], 4);
        if (KL >= 4)  result[v] += simd_shuffle_down(result[v], 2);
        if (KL >= 2)  result[v] += simd_shuffle_down(result[v], 1);
    }
    if (k_lane == 0 && out_row < N) {
        for (int v = 0; v < VT; v++) if (v < M) out[v * N + out_row] = static_cast<InT>(result[v]);
    }
"""

_kernel = mx.fast.metal_kernel(name="mlx_qmv_wide_port", input_names=["x", "w", "scales", "biases"],
                               output_names=["out"], source=SOURCE)


def qmv_wide(x2d, w, scales, biases, k_lanes=8):
    T, K = x2d.shape
    N = w.shape[0]
    rows_per_tg = (32 // k_lanes) * 2
    n_tg = (N + rows_per_tg - 1) // rows_per_tg
    return _kernel(inputs=[x2d, w, scales, biases], template=[("VT", T), ("KL", k_lanes), ("InT", x2d.dtype)],
                   grid=(n_tg * 64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(T, N)], output_dtypes=[x2d.dtype])[0]
