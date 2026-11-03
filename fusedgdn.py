"""Fused Gated-DeltaNet middle for Qwen3.5 decode/verify in MLX: one Metal dispatch per layer instead of ~12.

On the M1 Pro each small dependent kernel costs ~8-10 us; the GDN middle (conv1d+SiLU, q/k RMS-normalize,
gate math, recurrence, gated RMSNorm) is ~2.9 ms/token in situ. This kernel does all of it for T<=4 positions
(one threadgroup per value head, 1024 threads; the head's state stays in registers across positions) and writes
the per-position (conv, ssm) snapshots that capture-mode speculation needs.

Arithmetic mirrors stock MLX op for op (same dtypes, same formulas copied from MLX's unary/binary op structs and
rms_norm kernel, same lane partitioning and butterfly reduction as mlx_lm's gated-delta kernel), so outputs are
bit-identical to the unfused path; `python fusedgdn.py` checks that.

    import fusedgdn; fusedgdn.install(model)    # after the fp16 cast
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from mlx_lm.models import qwen3_5
import specmlx

HEADER = """
template <typename T> T mlx_sigmoid(T x) {          // MLX unary_ops.h Sigmoid
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
inline float mlx_log1p(float x) {                     // MLX utils.h log1p
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) return metal::numeric_limits<float>::max();
    if (xp1 == 1.0f) return x;
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}
template <typename T> T mlx_softplus(T x) {         // MLX binary_ops.h LogAddExp(x, 0)
    T y = T(0);
    if (metal::isnan(x)) return metal::numeric_limits<T>::quiet_NaN();
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf) ? maxval : (maxval + mlx_log1p(metal::exp(minval - maxval)));
}
inline float bfly(float v) {
    v += simd_shuffle_xor(v, 1); v += simd_shuffle_xor(v, 2); v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 8); v += simd_shuffle_xor(v, 16);
    return v;
}
"""

SOURCE = """
    const uint hv = threadgroup_position_in_grid.z;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint hk = hv / (HV / HK);
    constexpr uint KD = HK * DK, CD = 2 * KD + HV * DV, NP = DK / 32;
    threadgroup float qs[DK], ks[DK], vs[DV], outs[DV];
    threadgroup float gsh;
    threadgroup InT bsh;

    // this thread's conv channel (threads < 2*DK+DV) and whether it owns writing that channel's conv snapshots
    const bool has_c = tid < 2 * DK + DV;
    const uint c = tid < DK ? hk * DK + tid : (tid < 2 * DK ? KD + hk * DK + (tid - DK) : 2 * KD + hv * DV + (tid - 2 * DK));
    const bool owns_c = has_c && (tid >= 2 * DK || hv % (HV / HK) == 0);
    // recurrence registers: simdgroup sg owns state rows sg*4..sg*4+3, lane owns NP contiguous dk elements
    float st[DV / 32][NP];
    _Pragma("unroll") for (uint r = 0; r < DV / 32; r++) {
        const uint base = (hv * DV + sg * (DV / 32) + r) * DK + lane * NP;
        _Pragma("unroll") for (uint i = 0; i < NP; i++) st[r][i] = static_cast<float>(state_in[base + i]);
    }
    const float A = metal::precise::exp(static_cast<float>(A_log[hv]));

    for (uint t = 0; t < T; t++) {
        // conv window for position t = rows t..t+3 of concat(conv_state[3], qkv[T])
        #define ROW(j) ((t + (j)) < 3 ? conv_state[(t + (j)) * CD + c] : qkv[(t + (j) - 3) * CD + c])
        if (has_c) {
            float acc = 0.0f;
            _Pragma("unroll") for (uint j = 0; j < 4; j++) acc += static_cast<float>(ROW(j)) * conv_w[c * 4 + j];
            InT co = static_cast<InT>(acc);
            InT sv = co * mlx_sigmoid<InT>(co);
            if (tid < DK) qs[tid] = sv; else if (tid < 2 * DK) ks[tid - DK] = sv; else vs[tid - 2 * DK] = sv;
            if (owns_c) { _Pragma("unroll") for (uint j = 0; j < 3; j++) conv_snaps[(t * 3 + j) * CD + c] = ROW(j + 1); }
        }
        if (tid == 2 * DK + DV) {
            auto sp = mlx_softplus(a[t * HV + hv] + dt_bias[hv]);   // half+half or float promotion, as MLX
            gsh = metal::precise::exp((-A) * static_cast<float>(sp));
            bsh = mlx_sigmoid<InT>(b[t * HV + hv]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // normalize_qk: q = s2 * rms_norm(q), k = s1 * rms_norm(k), MLX rms_single_row arithmetic
        if (sg < 2) {
            threadgroup float* x = sg == 0 ? qs : ks;
            float acc = 0.0f;
            _Pragma("unroll") for (uint i = 0; i < NP; i++) { float xi = x[lane * NP + i]; acc += xi * xi; }
            acc = simd_sum(acc);
            const float inv = metal::precise::rsqrt(acc / DK + qk_eps[0]);
            const InT s = sg == 0 ? qk_scale[0] : qk_scale[1];
            float tmp[NP];
            _Pragma("unroll") for (uint i = 0; i < NP; i++) tmp[i] = static_cast<InT>(static_cast<InT>(1) * static_cast<InT>(x[lane * NP + i] * inv)) * s;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            _Pragma("unroll") for (uint i = 0; i < NP; i++) x[lane * NP + i] = tmp[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // recurrence (mlx_lm gated-delta kernel arithmetic)
        const float g = gsh;
        const float beta = static_cast<float>(bsh);
        _Pragma("unroll") for (uint r = 0; r < DV / 32; r++) {
            const uint dv = sg * (DV / 32) + r;
            float kv_mem = 0.0f;
            _Pragma("unroll") for (uint i = 0; i < NP; i++) { st[r][i] = st[r][i] * g; kv_mem += st[r][i] * ks[lane * NP + i]; }
            kv_mem = bfly(kv_mem);
            const float delta = (vs[dv] - kv_mem) * beta;
            float o = 0.0f;
            _Pragma("unroll") for (uint i = 0; i < NP; i++) { st[r][i] = st[r][i] + ks[lane * NP + i] * delta; o += st[r][i] * qs[lane * NP + i]; }
            o = bfly(o);
            if (lane == 0) outs[dv] = static_cast<InT>(o);
            const uint base = ((t * HV + hv) * DV + dv) * DK + lane * NP;
            _Pragma("unroll") for (uint i = 0; i < NP; i++) state_snaps[base + i] = st[r][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // gated RMSNorm: normed = w * InT(x * inv); y = InT(silu(float z) * float(normed))
        if (sg == 0) {
            float acc = 0.0f;
            _Pragma("unroll") for (uint i = 0; i < DV / 32; i++) { float xi = outs[lane * (DV / 32) + i]; acc += xi * xi; }
            acc = simd_sum(acc);
            const float inv = metal::precise::rsqrt(acc / DV + norm_eps[0]);
            _Pragma("unroll") for (uint i = 0; i < DV / 32; i++) {
                const uint j = lane * (DV / 32) + i;
                const InT normed = norm_w[j] * static_cast<InT>(outs[j] * inv);
                const float zf = static_cast<float>(z[(t * HV + hv) * DV + j]);
                const float gate = zf * mlx_sigmoid<float>(zf);
                y[(t * HV + hv) * DV + j] = static_cast<InT>(gate * static_cast<float>(normed));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="fused_gdn", input_names=["qkv", "conv_state", "conv_w", "z", "a", "b", "A_log", "dt_bias", "state_in",
                                   "norm_w", "qk_eps", "qk_scale", "norm_eps"],
    output_names=["y", "state_snaps", "conv_snaps"], header=HEADER, source=SOURCE)


def _consts(self, dtype):
    """Kernel constants, built eagerly (never inside an mx.compile trace). The fp16 scales are made the way MLX
    makes them for `python_float * fp16_array`, so they round identically."""
    inv_scale = self.head_k_dim ** -0.5
    one = mx.ones((1,), dtype)
    c = (mx.array([1e-6 * inv_scale ** 2], mx.float32), mx.concatenate([inv_scale ** 2 * one, inv_scale * one]),
         mx.array([self.norm.eps], mx.float32))
    mx.eval(c)
    return c


def fused(self, qkv, z, a, b, conv_state, state):
    """qkv [T, conv_dim], z [T, Hv*Dv], a/b [T, Hv], conv_state [(1,) 3, conv_dim], state [(1,) Hv, Dv, Dk] fp32
    -> y [T, Hv*Dv], state snapshots [T, Hv, Dv, Dk], conv snapshots [T, 3, conv_dim]."""
    T = qkv.shape[0]
    Hv, Hk, Dv, Dk = self.num_v_heads, self.num_k_heads, self.head_v_dim, self.head_k_dim
    if "_fused_consts" not in self.__dict__:
        self.__dict__["_fused_consts"] = _consts(self, qkv.dtype)
    qk_eps, qk_scale, norm_eps = self.__dict__["_fused_consts"]
    return _kernel(   # conv1d.weight [C, 4, 1] is read flat as [C, 4]
        inputs=[qkv, conv_state, self.conv1d.weight, z, a, b, self.A_log, self.dt_bias, state, self.norm.weight,
                qk_eps, qk_scale, norm_eps],
        template=[("InT", qkv.dtype), ("T", T), ("HV", Hv), ("HK", Hk), ("DV", Dv), ("DK", Dk)],
        grid=(1024, 1, Hv), threadgroup=(1024, 1, 1),
        output_shapes=[(T, Hv * Dv), (T, Hv, Dv, Dk), (T, 3, self.conv_dim)],
        output_dtypes=[qkv.dtype, mx.float32, conv_state.dtype])


_orig = qwen3_5.GatedDeltaNet.__call__


def _call(self, inputs, mask=None, cache=None):
    B, S, _ = inputs.shape
    if (cache is None or cache[0] is None or cache[1] is None or B != 1 or S > 4 or mask is not None
            or cache.lengths is not None or self.sharding_group is not None):
        return _orig(self, inputs, mask, cache)
    # never integer-index here: MLX lowers x[0] to a gather (a copy; ~50 us for the 2 MB state). Reshape/slices are views.
    qkv = self.in_proj_qkv(inputs).reshape(S, -1); z = self.in_proj_z(inputs).reshape(S, -1)
    b = self.in_proj_b(inputs).reshape(S, -1); a = self.in_proj_a(inputs).reshape(S, -1)
    y, ss, cs = fused(self, qkv, z, a, b, cache[0], cache[1])
    snaps = [(cs[t:t + 1], ss[t:t + 1]) for t in range(S)]
    cache[0], cache[1] = snaps[-1]
    if S > 1 and specmlx.CAPTURE:
        cache.snaps = snaps
    if S == 1:
        cache.advance(S)
    return self.out_proj(y.reshape(1, S, -1))


def install(model, on=True):
    """Patch (on) or restore (off) GatedDeltaNet; precompute constants."""
    for m in model.modules():
        if isinstance(m, qwen3_5.GatedDeltaNet) and "_fused_consts" not in m.__dict__:
            m.__dict__["_fused_consts"] = _consts(m, m.in_proj_z["scales"].dtype if "scales" in m.in_proj_z else m.in_proj_z.weight.dtype)
        if isinstance(m, qwen3_5.DecoderLayer):   # specfast's compiled verify graphs, kept per mode (cheap toggling)
            m.__dict__["_capture_fns"] = m.__dict__.setdefault("_capture_fns_fused" if on else "_capture_fns_stock", {})
    qwen3_5.GatedDeltaNet.__call__ = _call if on else _orig


if __name__ == "__main__":   # bit-exactness check against the stock path on a real layer and a real state
    from mlx.utils import tree_map
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    HERE = os.path.dirname(os.path.abspath(__file__))
    from bench import chat
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit")
    m, tok = load(path)
    m.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, m.parameters()))
    ids = tok.encode(chat("Prove that the square root of 2 is irrational."), add_special_tokens=False)
    cache = make_prompt_cache(m); m(mx.array([ids]), cache=cache); mx.eval([c.state for c in cache])
    Ls = m.language_model.model.layers
    bad = 0
    for li, L in enumerate(Ls):
        if not L.is_linear:
            continue
        G, c = L.linear_attn, cache[li]
        for T in (1, 2, 3, 4):
            x = (mx.random.normal((1, T, G.hidden_size)) * 0.5).astype(mx.float16)
            ref_c = type(c)(2); ref_c[0], ref_c[1] = c[0], c[1]
            new_c = type(c)(2); new_c[0], new_c[1] = c[0], c[1]
            specmlx.CAPTURE = True
            yr = _orig(G, x, None, ref_c)
            yf = _call(G, x, None, new_c)
            specmlx.CAPTURE = False
            ok = mx.array_equal(yr, yf).item() and mx.array_equal(ref_c[0], new_c[0]).item() and mx.array_equal(ref_c[1], new_c[1]).item()
            if T > 1:
                ok = ok and all(mx.array_equal(p, q).item() for sr, sf in zip(ref_c.snaps, new_c.snaps) for p, q in zip(sr, sf))
            if not ok:
                bad += 1
                print(f"layer {li} T={T}: MISMATCH  max|dy| {mx.abs(yr - yf).max().item():.3g}  "
                      f"max|dstate| {mx.abs(ref_c[1] - new_c[1]).max().item():.3g}  conv equal {mx.array_equal(ref_c[0], new_c[0]).item()}")
    print("FUSED_GDN bit-exact" if bad == 0 else f"FUSED_GDN {bad} mismatches")
