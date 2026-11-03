"""Score the model behind ./serve.sh (llama-server :8080) on GSM8K / MATH. Thinking off, greedy.

python evalmath.py TAG SET N [--ids ids.json]
SET: gsm_test | math_test (MATH-500) | gsm_dev (GSM8K train) | math_dev (MATH train/held-out, disjoint from MATH-500)
Writes results/TAG/SET.jsonl (resumable). Same seeded order for every model, so first-N subsets are paired.
"""
import argparse, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROMPT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def load(name):
    rows = []
    if name.startswith("gsm"):
        for l in open(f"{D}/gsm8k_{'test' if name == 'gsm_test' else 'train'}.jsonl"):
            r = json.loads(l)
            rows.append((r["question"], r["answer"].split("####")[-1].strip().replace(",", "")))
    else:
        f = "math500_test.jsonl" if name == "math_test" else "math_prm800k_train.jsonl"
        for l in open(f"{D}/{f}"):
            r = json.loads(l)
            rows.append((r["problem"], r["answer"]))
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    return [(i, *rows[i]) for i in idx]


def ask(port, q, max_tokens, prefix=""):
    # raw prompt = Qwen3.5 chat template with enable_thinking=false; identical for every model variant
    prompt = "<|im_start|>user\n" + PROMPT.format(q=q) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + prefix
    body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "seed": 0, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            return r["content"], ("length" if r.get("stop_type") == "limit" else "stop"), r.get("tokens_predicted", 0)
        except Exception as e:  # server restarting / transient
            if attempt == 4: raise
            print("retry", e, file=sys.stderr); time.sleep(10)


def grade(gold, text):
    from math_verify import parse, verify
    try:
        return bool(verify(parse("$" + gold + "$"), parse(text)))
    except Exception:
        return False


def main():
    a = argparse.ArgumentParser()
    a.add_argument("tag"); a.add_argument("set"); a.add_argument("n", type=int)
    a.add_argument("--port", type=int, default=8080); a.add_argument("--workers", type=int, default=16)
    a.add_argument("--ids", help="json list of problem ids to restrict to (after first-N)")
    a.add_argument("--max-tokens", type=int, default=2048)
    a.add_argument("--extend", type=int, help="continue answers truncated at --max-tokens up to this many tokens total")
    a = a.parse_args()
    if a.extend:
        return extend(a)
    probs = load(a.set)[: a.n]
    if a.ids:
        keep = set(json.load(open(a.ids))); probs = [p for p in probs if p[0] in keep]
    out = f"results/{a.tag}/{a.set}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [p for p in probs if p[0] not in done]
    t0, ntok = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        futs = [(p, ex.submit(ask, a.port, p[1], a.max_tokens)) for p in todo]
        for k, (p, fu) in enumerate(futs):
            text, fin, toks = fu.result()
            ntok += toks
            f.write(json.dumps({"id": p[0], "gold": p[2], "ok": grade(p[2], text), "finish": fin,
                                "toks": toks, "text": text}) + "\n"); f.flush()
            if (k + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"{k+1}/{len(todo)}  {ntok/dt:.1f} tok/s  {dt/60:.1f} min", file=sys.stderr)
    rs = [json.loads(l) for l in open(out)]
    keep = {p[0] for p in probs}
    rs = [r for r in rs if r["id"] in keep]
    acc = sum(r["ok"] for r in rs) / len(rs)
    trunc = sum(r["finish"] == "length" for r in rs)
    print(f"{a.tag} {a.set} n={len(rs)} acc={acc:.4f} truncated={trunc} mean_toks={sum(r['toks'] for r in rs)/len(rs):.0f}")


def extend(a):
    """Greedy continuation of truncated answers == re-running with a bigger cap, at a fraction of the cost.
    Keeps the cap-2048 verdict in ok2048. Uses <=8 workers so long sequences fit the shared 40960-token KV cache."""
    out = f"results/{a.tag}/{a.set}.jsonl"
    rs = [json.loads(l) for l in open(out)]
    keep = {p[0] for p in load(a.set)[: a.n]}
    q = {p[0]: p[1] for p in load(a.set)}
    todo = [r for r in rs if r["id"] in keep and r["finish"] == "length" and r["toks"] < a.extend]
    with ThreadPoolExecutor(min(a.workers, 8)) as ex:
        futs = [(r, ex.submit(ask, a.port, q[r["id"]], a.extend - r["toks"], r["text"])) for r in todo]
        for r, fu in futs:
            text, fin, toks = fu.result()
            r.setdefault("ok2048", r["ok"])
            r["text"] += text; r["toks"] += toks; r["finish"] = fin; r["ok"] = grade(r["gold"], r["text"])
    for r in rs:
        r.setdefault("ok2048", r["ok"])
    with open(out + ".tmp", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rs)
    os.replace(out + ".tmp", out)
    rs = [r for r in rs if r["id"] in keep]
    print(f"{a.tag} {a.set} n={len(rs)} acc@{a.extend}={sum(r['ok'] for r in rs)/len(rs):.4f} "
          f"acc@{a.max_tokens}={sum(r['ok2048'] for r in rs)/len(rs):.4f} still_truncated={sum(r['finish']=='length' for r in rs)} extended={len(todo)}")


if __name__ == "__main__":
    main()
