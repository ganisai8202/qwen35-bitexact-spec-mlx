"""Paired A/B of the fused Gated-DeltaNet kernel (fusedgdn.py), 9B 4-bit, 20-prompt suite, GPU otherwise idle.
Per prompt and mode, stock and fused run back to back (order alternating), ROUNDS times. Reports mean tok/s, the
mean per-prompt speed ratio with a bootstrap 95% CI, and token identity of fused vs stock plain greedy.
-> results/fused_gdn.json"""
import json, os, random, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
import fusedgdn, qmvt, specfast, specmtp
from specmlx import EOS, plain
from specpipe import spec_pipe
from bench import thermal_state

model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit")
mtp_path = sys.argv[2] if len(sys.argv) > 2 else None
N, ROUNDS = 256, 2
suite = json.load(open(os.path.join(HERE, "prompts_suite.json")))
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
qmvt.install(); specfast.install()
target, tok = load(model)
target.update(tree_map(lambda q: q.astype(mx.float16) if q.dtype == mx.bfloat16 else q, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
lm = target.language_model
mtp = specmtp.load_mtp(lm.args, mx.float16, mtp_path)
head = specmtp.Head(specmtp.head_module(lm), specmtp.frequent_vocab(tok, 65536), specmtp.full_head_fn(lm))
P = [tok.encode(chat(q), add_special_tokens=False) for qs in suite.values() for q in qs]
run = {"plain": lambda ids: plain(target, ids, N)[:2], "exact-spec": lambda ids: spec_pipe(target, mtp, head, ids, N, 0, 2)[:2]}
for on in (False, True):   # warm both modes (kernels, compiled verify graphs for T=2,3)
    fusedgdn.install(target, on)
    for ids in P[:3]:
        run["plain"](ids); run["exact-spec"](ids)

tps = {(m, on): [[] for _ in P] for m in run for on in (False, True)}
ident = []
for rnd in range(ROUNDS):
    for i, ids in enumerate(P):
        for mode in run:
            ref = None
            for on in ((False, True) if (i + rnd) % 2 == 0 else (True, False)):
                fusedgdn.install(target, on)
                out, t = run[mode](ids)
                tps[(mode, on)][i].append(t)
                if mode == "plain" and not on:
                    ref = out
                if mode == "plain" and on:
                    fused_out = out
            ident.append(fused_out == ref) if mode == "plain" else None
    print(f"round {rnd} done, thermal {thermal_state()}", flush=True)

rng = random.Random(0)
summary = {}
for mode in run:
    s = [statistics.mean(v) for v in tps[(mode, False)]]; f = [statistics.mean(v) for v in tps[(mode, True)]]
    ratios = [b / a for a, b in zip(s, f)]
    boots = sorted(statistics.mean(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    summary[mode] = {"stock": round(statistics.mean(s), 2), "fused": round(statistics.mean(f), 2),
                     "ratio": round(statistics.mean(ratios), 4), "ci95": [round(boots[250], 4), round(boots[9750], 4)]}
    print(f"FUSED {mode}: stock {summary[mode]['stock']:.2f} -> fused {summary[mode]['fused']:.2f} tok/s, "
          f"paired ratio {summary[mode]['ratio']:.3f} [{summary[mode]['ci95'][0]:.3f}, {summary[mode]['ci95'][1]:.3f}]", flush=True)
print(f"FUSED plain identical to stock plain greedy: {sum(ident)}/{len(ident)}")
json.dump({"summary": summary, "identical": [sum(ident), len(ident)],
           "per_prompt": {f"{m} {'fused' if on else 'stock'}": v for (m, on), v in tps.items()}},
          open(os.path.join(HERE, "results/fused_gdn.json"), "w"), indent=1)
