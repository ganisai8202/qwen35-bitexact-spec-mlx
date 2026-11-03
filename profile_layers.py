"""Per-component cost at T=1 vs T=5 tokens (ms, median), bf16 model as shipped vs cast to fp16 (M1 has no native bf16).
Components: one DeltaNet decoder layer, one full-attention decoder layer, an MLP alone, final norm + lm_head."""
import os, statistics, sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench import cooldown


def t_of(fn, reps=15):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 3)


def profile(model, tag):
    lm = model.language_model
    layers = lm.model.layers
    caches = make_prompt_cache(model)
    dt = lm.model.embed_tokens.scales.dtype if hasattr(lm.model.embed_tokens, "scales") else mx.float16
    ctx = mx.random.normal((1, 64, 4096)).astype(dt)
    for li in (0, 3):  # give the caches some history so T>1 is a real continuation
        layers[li](ctx, mask="causal" if not layers[li].is_linear else None, cache=caches[li])
    mx.eval([c.state for c in caches])
    out = {}
    for T in (1, 5):
        h = mx.random.normal((1, T, 4096)).astype(dt)
        def run_layer(li):
            c = caches[li]
            def f():
                r = layers[li](h, mask=("causal" if T > 1 else None) if not layers[li].is_linear else None, cache=c)
                if hasattr(c, "trim") and c.is_trimmable(): c.trim(T)
                return r
            return f
        st = caches[0].state
        out[f"deltanet T={T}"] = t_of(run_layer(0)); caches[0].state = st
        out[f"attention T={T}"] = t_of(run_layer(3))
        out[f"mlp T={T}"] = t_of(lambda: layers[0].mlp(h))
        out[f"norm+lm_head T={T}"] = t_of(lambda: lm.lm_head(lm.model.norm(h)))
    est = {T: 24 * out[f"deltanet T={T}"] + 8 * out[f"attention T={T}"] + out[f"norm+lm_head T={T}"] for T in (1, 5)}
    print(tag, out, "| est. full forward ms:", {T: round(v, 1) for T, v in est.items()}, flush=True)


model, _ = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
cooldown(30)
profile(model, "bf16")
# cast every floating-point parameter (scales, biases, norms, conv, A_log...) to fp16; packed uint32 weights untouched
model.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, model.parameters()))
profile(model, "fp16")
