"""Layer-by-layer timing of the REAL forward path (same masks/caches as Qwen3_5TextModel) at T=1 vs T=5.
A sync after every layer adds the same constant to both, so per-layer differences are meaningful."""
import os, statistics, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench import chat, cooldown

model, tok = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
lm = model.language_model; tm = lm.model
ids = tok.encode(chat("Explain photosynthesis briefly."), add_special_tokens=False)
cache = make_prompt_cache(model)
model(mx.array([ids]), cache=cache); mx.eval([c.state for c in cache])
cooldown(20)


def forward_timed(T):
    t = {"embed": 0.0, "deltanet": 0.0, "attention": 0.0, "head": 0.0}
    snap = [c.state for c in cache]
    t0 = time.perf_counter()
    h = tm.embed_tokens(mx.array([[ids[-1]] * T])); mx.eval(h)
    t["embed"] += time.perf_counter() - t0
    fa_mask = create_attention_mask(h, cache[tm.fa_idx]); ssm_mask = create_ssm_mask(h, cache[tm.ssm_idx])
    for layer, c in zip(tm.layers, cache):
        t0 = time.perf_counter()
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c); mx.eval(h)
        t["deltanet" if layer.is_linear else "attention"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    mx.eval(lm.lm_head(tm.norm(h)))
    t["head"] += time.perf_counter() - t0
    for c, s in zip(cache, snap):  # restore so every repetition starts from the same context
        c.state = s
    return t


for T in (1, 5):
    runs = [forward_timed(T) for _ in range(12)]
    med = {k: round(statistics.median(r[k] for r in runs) * 1e3, 2) for k in runs[0]}
    print(f"T={T}", med, "total", round(sum(med.values()), 1), flush=True)
