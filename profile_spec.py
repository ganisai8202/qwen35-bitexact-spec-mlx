"""Where does a speculative cycle's time go? Times (ms, median of 15, synced) of:
plain target step (T=1), target forward with T=2..8 (normal and DeltaNet-capture), and a draft step (T=1)."""
import os, statistics, sys, time
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import specmlx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from bench import chat, cooldown


def timed(fn, reps=15):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def bench_model(model, ids, Ts, capture_too=False):
    cache = make_prompt_cache(model)
    model(mx.array([ids]), cache=cache); mx.eval([c.state for c in cache])
    out = {}
    for T in Ts:
        for cap in ([False, True] if capture_too and T > 1 else [False]):
            def step():
                specmlx.CAPTURE = cap
                r = model(mx.array([[ids[-1]] * T]), cache=cache)
                specmlx.CAPTURE = False
                for c in cache:  # keep the cache length fixed
                    if hasattr(c, "trim") and c.is_trimmable(): c.trim(T)
                return r
            out[f"T={T}{' capture' if cap else ''}"] = round(timed(step), 2)
    return out


target, tok = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
draft, _ = load(os.path.join(HERE, "models/Qwen3.5-0.8B-4bit"))
ids = tok.encode(chat("Explain photosynthesis briefly."), add_special_tokens=False)
cooldown(30)
print("target", bench_model(target, ids, [1, 2, 3, 5, 8], capture_too=True))
print("draft ", bench_model(draft, ids, [1, 2]))
