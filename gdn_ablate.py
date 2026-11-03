"""Which phase of the fused GDN kernel is slow? Time variants with phases removed (outputs are garbage; timing only)."""
import os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import fusedgdn
from gdn_micro import G, Hv, Dv, CD

S = fusedgdn.SOURCE
cut = lambda marker: S.replace(marker, "return;\n" + marker, 1)
variants = {
    "empty": "if (thread_position_in_threadgroup.x == 0) y[threadgroup_position_in_grid.z] = 0;",
    "full": S,
    "no-state-io": S.replace("st[r][i] = static_cast<float>(state_in[base + i]);", "st[r][i] = 0.1f;")
                    .replace("state_snaps[base + i] = st[r][i];", "if (st[r][i] == 12345.f) state_snaps[base + i] = 0;"),
    "stop-before-conv": cut("    for (uint t = 0; t < T; t++) {"),
    "stop-after-recurrence": cut("        // gated RMSNorm"),
}
qkv = mx.random.normal((1, CD)).astype(mx.float16); z = mx.random.normal((1, Hv * Dv)).astype(mx.float16)
a = mx.random.normal((1, Hv)).astype(mx.float16); b = mx.random.normal((1, Hv)).astype(mx.float16)
conv = mx.random.normal((3, CD)).astype(mx.float16); st0 = mx.random.normal((Hv, Dv, 128)) * 0.1
mx.eval(qkv, z, a, b, conv, st0)
for tg in (1024,):
    for name, src in variants.items():
        k = mx.fast.metal_kernel(name="abl_" + name.replace("-", "_"), input_names=fusedgdn._kernel_inputs if hasattr(fusedgdn, "_kernel_inputs") else
                                 ["qkv", "conv_state", "conv_w", "z", "a", "b", "A_log", "dt_bias", "state_in", "norm_w", "qk_eps", "qk_scale", "norm_eps"],
                                 output_names=["y", "state_snaps", "conv_snaps"], header=fusedgdn.HEADER, source=src)
        fusedgdn._kernel, orig = k, fusedgdn._kernel
        ts = []
        for _ in range(12):
            s, c = st0, conv
            t0 = time.perf_counter()
            for _ in range(24):
                y, ss, cs = fusedgdn.fused(G, qkv, z, a, b, c, s)
                s, c = ss, cs
            mx.eval(y, s, c)
            ts.append((time.perf_counter() - t0) * 1e6 / 24)
        fusedgdn._kernel = orig
        print(f"ABLATE {name:<20} {statistics.median(ts[3:]):7.1f} us/layer", flush=True)
