"""Quality + speed of the optimized decoder (default bit-exact mode) on the same dev problems the Q4_K_M baseline
was scored on: same prompt, thinking off, greedy, 2048-token cap, math-verify grader. Exact mode is byte-identical
to plain greedy MLX decoding, so this scores exactly what fastqwen.py produces. Also logs per-problem tok/s.

python quality_spec.py SET N        (SET: math_dev | gsm_dev)  -> ../surgery/results/mlx_spec/SET.jsonl (resumable)
then: cd ../surgery && KEY=ok2048 .venv/bin/python -I compare.py 100 math_dev,gsm_dev base mlx_spec
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SURGERY = os.environ.get("SURGERY", os.path.join(HERE, "../surgery")); sys.path.insert(0, SURGERY)   # evalmath.py + data/ + MLX model dir
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
import qmvt, specfast, specmtp
from specmlx import EOS
from specpipe import spec_pipe
from evalmath import load as load_set, PROMPT, grade

S, N = sys.argv[1], int(sys.argv[2])
EXT = int(sys.argv[sys.argv.index("--extend") + 1]) if "--extend" in sys.argv else None
qmvt.install(); specfast.install()
mx.set_cache_limit(1 << 30)   # long runs: MLX keeps freed buffers cached; uncapped it grew to 10 GB and swapped
target, tok = load(os.path.join(SURGERY, "mlx/qwen35-9b-4bit"))
target.update(tree_map(lambda q: q.astype(mx.float16) if q.dtype == mx.bfloat16 else q, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
mtp = specmtp.load_mtp(target.language_model.args, mx.float16)
head = specmtp.Head(specmtp.head_module(target.language_model), specmtp.frequent_vocab(tok, 65536), specmtp.full_head_fn(target.language_model))
spec_pipe(target, mtp, head, tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False), 16, 1)
out_f = os.path.join(SURGERY, f"results/mlx_spec/{S}.jsonl")
if EXT:  # continue answers truncated at 2048 up to EXT tokens (same method as the baseline's --extend); keep ok2048
    qs = {pid: q for pid, q, _ in load_set(S)}
    rs = [json.loads(l) for l in open(out_f)]
    for r in rs:
        if r["finish"] == "length" and r["toks"] < EXT:
            ids = tok.encode("<|im_start|>user\n" + PROMPT.format(q=qs[r["id"]]) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             add_special_tokens=False) + tok.encode(r["text"], add_special_tokens=False)
            out, tps, acc, tpc = spec_pipe(target, mtp, head, ids, EXT - r["toks"], 0, 2)
            r["text"] += tok.decode([t for t in out if t not in EOS]); r["toks"] += len(out)
            r["finish"] = "stop" if EOS & set(out) else "length"; r["ok"] = grade(r["gold"], r["text"])
    with open(out_f + ".tmp", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rs)
    keep = out_f.replace(".jsonl", ".cap2048.jsonl")
    if not os.path.exists(keep):   # keep the first (pristine 2048-cap) file; a re-run must not overwrite it
        os.replace(out_f, keep)
    os.replace(out_f + ".tmp", out_f)
    print(f"EXTENDED {S}: acc@{EXT}={sum(r['ok'] for r in rs[:N]) / len(rs[:N]):.3f} acc@2048={sum(r['ok2048'] for r in rs[:N]) / len(rs[:N]):.3f} "
          f"still truncated={sum(r['finish'] == 'length' for r in rs[:N])}", flush=True)
    sys.exit(0)
os.makedirs(os.path.dirname(out_f), exist_ok=True)
done = {json.loads(l)["id"] for l in open(out_f)} if os.path.exists(out_f) else set()
t_all, n_tok = time.time(), 0
with open(out_f, "a") as f:
    for k, (pid, q, gold) in enumerate(load_set(S)[:N]):
        if pid in done:
            continue
        ids = tok.encode("<|im_start|>user\n" + PROMPT.format(q=q) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                         add_special_tokens=False)
        out, tps, acc, tpc = spec_pipe(target, mtp, head, ids, 2048, 0, 2)
        text = tok.decode([t for t in out if t not in EOS])
        ok = grade(gold, text)
        hit_eos = bool(EOS & set(out))
        f.write(json.dumps({"id": pid, "gold": gold, "ok": ok, "ok2048": ok, "toks": len(out), "finish": "stop" if hit_eos else "length",
                            "tps": round(tps, 2), "accept": round(acc, 3), "tok_per_cycle": round(tpc, 3), "text": text}) + "\n")
        f.flush(); n_tok += len(out)
        if (k + 1) % 10 == 0:
            print(f"{S} {k + 1}/{N}  {n_tok / (time.time() - t_all):.1f} tok/s overall", flush=True)
rs = [json.loads(l) for l in open(out_f)][:N]
print(f"DONE {S} n={len(rs)} acc@2048={sum(r['ok'] for r in rs) / len(rs):.3f} "
      f"mean tok/s={sum(r['tps'] for r in rs) / len(rs):.1f} truncated={sum(r['finish'] == 'length' for r in rs)}", flush=True)
