"""Fuse each Gated-DeltaNet decoder layer's single-token step with mx.compile.

In decode, a DeltaNet layer is a pure function with fixed shapes:
(x [1,1,4096], conv_state, ssm_state) -> (x', conv_state', ssm_state'). Unlike attention, whose KV cache grows,
it can be compiled once. Compiling fuses the dozens of small element-wise kernels (gating, norms, conv taps,
activations) per layer into a few. Attention layers are left as they are.

    import fastdecode; fastdecode.install()
"""
import mlx.core as mx
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache

_orig = qwen3_5.DecoderLayer.__call__


def _call(self, x, mask=None, cache=None):
    if not (self.is_linear and cache is not None and x.shape[1] == 1 and cache[0] is not None and cache.lengths is None):
        return _orig(self, x, mask, cache)
    fn = self.__dict__.get("_compiled_step")
    if fn is None:
        def step(x, conv_state, ssm_state):
            c = ArraysCache(2)
            c[0], c[1] = conv_state, ssm_state
            out = _orig(self, x, None, c)
            return out, c[0], c[1]
        fn = mx.compile(step)
        self.__dict__["_compiled_step"] = fn
    out, cache[0], cache[1] = fn(x, cache[0], cache[1])
    return out


def install():
    qwen3_5.DecoderLayer.__call__ = _call


def uninstall():
    qwen3_5.DecoderLayer.__call__ = _orig
