"""Cut the CPU graph-build time of speculative verification (the GPU idles during it, since a cycle syncs once).

In capture mode (specmlx.CAPTURE), a DeltaNet decoder layer at T=2..4 is a fixed-shape pure function
(x, conv_state, ssm_state) -> (out, final states, per-position snapshots), so each (layer, T) is compiled once
with mx.compile. Later calls replay the cached graph instead of rebuilding ~100 small ops in Python.
Attention layers (growing KV) are left as they are.

    import specfast; specfast.install()
"""
import mlx.core as mx
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache
import specmlx

_orig = qwen3_5.DecoderLayer.__call__


def _call(self, x, mask=None, cache=None):
    T = x.shape[1]
    if not (specmlx.CAPTURE and self.is_linear and 2 <= T <= 4 and cache is not None and cache[0] is not None):
        return _orig(self, x, mask, cache)
    fns = self.__dict__.setdefault("_capture_fns", {})
    fn = fns.get(T)
    if fn is None:
        def step(x, conv_state, ssm_state):
            c = ArraysCache(2)
            c[0], c[1] = conv_state, ssm_state
            out = _orig(self, x, None, c)
            flat = [a for pair in c.snaps for a in pair]
            return [out, c[0], c[1]] + flat
        fn = fns[T] = mx.compile(step)
    res = fn(x, cache[0], cache[1])
    out, cache[0], cache[1] = res[0], res[1], res[2]
    cache.snaps = [(res[3 + 2 * j], res[4 + 2 * j]) for j in range(T)]
    return out


def install():
    qwen3_5.DecoderLayer.__call__ = _call
