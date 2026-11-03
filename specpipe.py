"""MTP self-speculation with optimistic prebuilding: build cycle i+1's graph while the GPU verifies cycle i.

A cycle syncs once, to count accepted drafts. Without this, the GPU idles while Python builds the next graph
(~2.3 ms of a ~36 ms k=1 cycle). Here, right after cycle i is launched, the next cycle is built lazily assuming
every draft is accepted. That is the common case, and it needs no rewind: the caches are already in the
post-verify state, and the bonus token, drafts and target hiddens are all lazy arrays. After the sync:
  * guess right -> launch the prebuilt graph at once (no bubble);
  * guess wrong -> drop the unlaunched graph (no GPU work wasted), restore cache references, rewind as usual.
Output is identical to specmtp.spec_mtp (same drafts, same verification).

    from specpipe import spec_pipe   # same signature/return as specmtp.spec_mtp
"""
import time
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
import specmlx
import specmtp
from specmlx import EOS


def _kv_snap(c):
    # MLX KV writes are in place (the Python object is rebound to a new graph node), so keep a zero-copy
    # view of the CURRENT node. Restoring it drops any pending write from a discarded prebuild.
    return (c.keys[...] if c.keys is not None else None, c.values[...] if c.values is not None else None, c.offset)


def _kv_restore(c, s):
    c.keys, c.values, c.offset = s


def _snapshot(tc, mc):
    t = []
    for c in tc:
        t.append((c[0], c[1], getattr(c, "snaps", None)) if isinstance(c, ArraysCache) else _kv_snap(c))
    return t, _kv_snap(mc)


def _restore(tc, mc, snap):
    t, ms = snap
    for c, s in zip(tc, t):
        if isinstance(c, ArraysCache):
            c[0], c[1] = s[0], s[1]
            c.snaps = s[2]
        else:
            _kv_restore(c, s)
    _kv_restore(mc, ms)


def spec_pipe(target, mtp, head, ids, n, k, kmax=3, on_tokens=None):
    lm = target.language_model
    body, full_head, embed = lm.model, specmtp.full_head_fn(lm), lm.model.embed_tokens
    tc, mc = make_prompt_cache(target), KVCache()
    hp = body(mx.array([ids[:-1]]), cache=tc)
    hs = [hp[:, i:i + 1] for i in range(hp.shape[1])]
    if len(ids) > 2:
        mtp(mx.concatenate(hs[:-1], axis=1), embed(mx.array([ids[1:-1]])), mc)
    mx.eval([c.state for c in tc] + [mc.state])
    tok = [mx.array(t) for t in ids]           # token per position (lazy-capable)
    adaptive = k == 0
    st = {"kc": 2 if adaptive else k, "ema": 0.85}

    def next_k(st, k, i):
        if not adaptive:
            return st["kc"], st["ema"]
        ema = 0.9 * st["ema"] + 0.1 * (i >= 1)
        kcap = 1 if ema < 0.8 else (min(2, kmax) if ema < 0.9 else kmax)
        return (min(k + 1, kcap) if i == k else max(1, min(i + 1, kcap))), ema

    def build(c, m, k):
        q = list(range(m + 1, c + 1))
        o = mtp(mx.concatenate([hs[p - 1] for p in q], axis=1), embed(mx.stack([tok[p] for p in q])[None]), mc)
        d = [head.argmax(o)]
        for _ in range(k - 1):
            o = mtp(o[:, -1:], embed(d[-1].reshape(1, 1)), mc)
            d.append(head.argmax(o))
        specmlx.CAPTURE = True
        hv = body(mx.stack([tok[c]] + d)[None], cache=tc)
        specmlx.CAPTURE = False
        g = mx.argmax(full_head(hv), axis=-1).reshape(-1)
        return d, g, hv

    out, cycles, accepted, drafted, hits = [], 0, 0, 0, 0
    c, m, k = len(ids) - 1, len(ids) - 2, st["kc"]
    t0 = time.perf_counter()
    d, g, hv = build(c, m, k)
    D = mx.stack(d)
    mx.async_eval(D, g)
    while True:
        # --- optimistic prebuild of the next cycle (assume all k drafts accepted)
        pre = _snapshot(tc, mc)
        n_tok, n_hs = len(tok), len(hs)
        tok += [D[j] for j in range(k)] + [g[k]]
        hs += [hv[:, j:j + 1] for j in range(k + 1)]
        mc.trim(k - 1)
        k2, ema2 = next_k(st, k, k)
        d2, g2, hv2 = build(c + k + 1, c, k2)
        # --- sync on the current cycle
        dl, gl = D.tolist(), g.tolist()
        i = 0
        while i < k and dl[i] == gl[i]:
            i += 1
        new = dl[:i] + [gl[i]]
        out += new; cycles += 1; accepted += i; drafted += k
        if on_tokens:
            on_tokens(new)
        done = len(out) >= n or bool(EOS & set(new))
        if i == k:                      # guess right: launch prebuilt graph immediately
            hits += 1
            st["kc"], st["ema"] = k2, ema2
            if done:
                break
            c, m, k = c + k + 1, c, k2
            d, g, hv = d2, g2, hv2
            D = mx.stack(d)
            mx.async_eval(D, g)
            continue
        # guess wrong: discard prebuilt graph, restore, rewind cycle i to i accepted
        del tok[n_tok:]; del hs[n_hs:]
        _restore(tc, mc, pre)
        tok += [mx.array(t) for t in new]
        hs += [hv[:, j:j + 1] for j in range(i + 1)]
        specmlx.rewind(tc, k - i, snap_idx=i)
        if k > 1:
            mc.trim(k - 1)
        st["kc"], st["ema"] = next_k(st, k, i)
        if done:
            break
        c, m, k = c + i + 1, c, st["kc"]
        d, g, hv = build(c, m, k)
        D = mx.stack(d)
        mx.async_eval(D, g)
    dt = time.perf_counter() - t0
    if EOS & set(out):
        out = out[:min(out.index(e) for e in EOS if e in out) + 1]
    spec_pipe.last_hit_rate = hits / cycles
    return out[:n], len(out) / dt, accepted / drafted, len(out) / cycles
