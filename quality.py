"""Accuracy check for an MLX variant on the same 100 GSM8K-train + 100 MATH dev problems the Q4_K_M baseline was
scored on (../surgery/results/base). Same prompt, same chat format (thinking off), greedy, 2048-token cap, same
math-verify grader. Writes ../surgery/results/TAG/{gsm_dev,math_dev}.jsonl so ../surgery/compare.py pairs it.

python quality.py TAG [--model PATH] [--fp16] [--batch 24]
then: cd ../surgery && KEY=ok2048 python compare.py 100 gsm_dev,math_dev base TAG
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SURGERY = os.path.join(HERE, "../surgery")
sys.path.insert(0, SURGERY)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load, batch_generate
from evalmath import load as load_set, PROMPT, grade

a = argparse.ArgumentParser()
a.add_argument("tag"); a.add_argument("--model", default=os.path.join(SURGERY, "mlx/qwen35-9b-4bit"))
a.add_argument("--fp16", action="store_true"); a.add_argument("--batch", type=int, default=24)
a.add_argument("-n", type=int, default=100)
a = a.parse_args()
model, tok = load(a.model)
if a.fp16:
    model.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, model.parameters()))
for s in ("gsm_dev", "math_dev"):
    probs = load_set(s)[:a.n]
    prompts = [tok.encode("<|im_start|>user\n" + PROMPT.format(q=q) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          add_special_tokens=False) for _, q, _ in probs]
    t0 = time.time()
    r = batch_generate(model, tok, prompts, max_tokens=2048, return_token_ids=True,
                       completion_batch_size=a.batch, prefill_batch_size=8)
    dt = time.time() - t0
    out = os.path.join(SURGERY, f"results/{a.tag}/{s}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ntok, nok = 0, 0
    with open(out, "w") as f:
        for (pid, _, gold), text, ids in zip(probs, r.texts, r.token_ids):
            ok = grade(gold, text); ntok += len(ids); nok += ok
            f.write(json.dumps({"id": pid, "gold": gold, "ok": ok, "ok2048": ok, "toks": len(ids),
                                "finish": "length" if len(ids) >= 2048 else "stop", "text": text}) + "\n")
    print(f"{a.tag} {s} n={len(probs)} acc@2048={nok / len(probs):.3f} tokens={ntok} "
          f"aggregate={ntok / dt:.1f} tok/s ({dt / 60:.1f} min)", flush=True)
