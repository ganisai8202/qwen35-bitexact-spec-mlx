"""Lossless projection fusion for Qwen3.5 in MLX: one matmul instead of 2-4 when projections share an input.

DeltaNet: in_proj_qkv + in_proj_z + in_proj_b + in_proj_a (the last two have only 32 outputs each, so as separate
kernels they are almost pure launch overhead). MLP: gate_proj + up_proj. Attention: q_proj + k_proj + v_proj.
Affine 4-bit quantization is per row (weight/scales/biases are row-wise), so concatenating rows is exact.
Each original projection is replaced by a proxy: the first call with a given input array runs the fused matmul
and caches the split outputs, and the sibling projections called with the same array object reuse them.
Originals are dropped, so there are no duplicate weights in memory.

    import fuse; fuse.apply(model)
"""
import mlx.core as mx
import mlx.nn as nn


class _Group:
    def __init__(self, layers):
        self.lin = nn.QuantizedLinear(layers[0]["weight"].shape[1] * 8, sum(l["weight"].shape[0] for l in layers),
                                      bias=False, group_size=64, bits=4)
        self.lin.weight = mx.concatenate([l["weight"] for l in layers])
        self.lin.scales = mx.concatenate([l["scales"] for l in layers])
        self.lin.biases = mx.concatenate([l["biases"] for l in layers])
        sizes = [l["weight"].shape[0] for l in layers]
        self.cuts = [sum(sizes[:i + 1]) for i in range(len(sizes) - 1)]
        self.x, self.outs = None, None

    def get(self, x, i):
        if self.x is not x:
            self.outs = mx.split(self.lin(x), self.cuts, axis=-1)
            self.x = x
        return self.outs[i]


class _Proxy:
    def __init__(self, group, i):
        self.group, self.i = group, i

    def __call__(self, x):
        return self.group.get(x, self.i)


def _fuse(module, names):
    layers = [getattr(module, n) for n in names]
    if not all(isinstance(l, nn.QuantizedLinear) and l.bits == 4 and l.group_size == 64 and "bias" not in l for l in layers):
        return 0
    g = _Group(layers)
    for i, n in enumerate(names):
        delattr(module, n) if n in module else None
        object.__setattr__(module, n, _Proxy(g, i))
    mx.eval(g.lin.parameters())
    return 1


def apply(model):
    n = 0
    for layer in model.language_model.model.layers:
        if layer.is_linear:
            n += _fuse(layer.linear_attn, ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"])
        else:
            n += _fuse(layer.self_attn, ["q_proj", "k_proj", "v_proj"])
        n += _fuse(layer.mlp, ["gate_proj", "up_proj"])
    return n
