"""Paper benchmark: fixed 20-prompt suite (math/code/prose/chat), thinking off, greedy, 256 new tokens.
Rounds rotate configuration order to cancel drift; reports per-config mean tok/s with bootstrap 95% CI over
prompts, per-category means, thermal state, and token-identity of exact mode vs plain greedy.

python final_bench.py --model PATH --mtp PATH --tag 9b [--rounds 3] [--llamacpp]   -> results/final_<tag>.json
"""
import argparse, json, os, random, statistics, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load, stream_generate
import qmvt, specfast, specmtp
from specmlx import EOS, plain
from specpipe import spec_pipe
from bench import SERVER, GGUF, post, thermal_state

a = argparse.ArgumentParser()
a.add_argument("--model", default=os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit")); a.add_argument("--mtp", default=None)
a.add_argument("--fused-gdn", action="store_true", help="ours + fused DeltaNet kernel (fusedgdn.py)")
a.add_argument("--tag", default="9b"); a.add_argument("--rounds", type=int, default=3); a.add_argument("-n", type=int, default=256)
a.add_argument("--llamacpp", action="store_true", help="also run llama.cpp (Ollama b10969) on --gguf (+ native MTP if --mtp-gguf)")
a.add_argument("--gguf", default=GGUF); a.add_argument("--mtp-gguf", default=os.path.join(HERE, "models/mtp_gguf/Qwen3.5-9B-MTP-ONLY-Q4_K_M.gguf"))
a = a.parse_args()
suite = json.load(open(os.path.join(HERE, "prompts_suite.json")))
P = [(cat, q) for cat, qs in suite.items() for q in qs]
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

qmvt.install(); specfast.install()
target, tok = load(a.model)
target.update(tree_map(lambda q: q.astype(mx.float16) if q.dtype == mx.bfloat16 else q, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
lm = target.language_model
if a.fused_gdn:   # stock mlx-lm row stays stock: the patch is switched off around it below
    import fusedgdn
    fusedgdn.install(target, False)
mtp = specmtp.load_mtp(lm.args, mx.float16, a.mtp)
head = specmtp.Head(specmtp.head_module(lm), specmtp.frequent_vocab(tok, 65536), specmtp.full_head_fn(lm))
warm = tok.encode(chat("Count from one to twenty."), add_special_tokens=False)
for on in ([False, True] if a.fused_gdn else [False]):   # compile kernels and verify graphs (T=2..4) for each mode
    if a.fused_gdn:
        fusedgdn.install(target, on)
    plain(target, warm, 8)
    for k in (1, 2, 3):
        spec_pipe(target, mtp, head, warm, 24, k)


def gdn(on):
    if a.fused_gdn:
        fusedgdn.install(target, on)


def mlx_lm_plain(ids):  # the stock mlx-lm generator (what users run today)
    gdn(False)
    last = None
    for r in stream_generate(target, tok, ids, max_tokens=a.n):
        last = r
    return last.generation_tps, None


def ours(kmax):
    def run(ids):
        gdn(True)
        out, tps, acc, tpc = spec_pipe(target, mtp, head, ids, a.n, 0, kmax)
        return tps, out
    return run


configs = {"mlx-lm stock": mlx_lm_plain, "ours exact (k<=2)": ours(2), "ours fast (k<=3)": ours(3)}
res = {c: [[] for _ in P] for c in configs}
same = []
refs = {}
for rnd in range(a.rounds):
    order = list(configs)[rnd % len(configs):] + list(configs)[:rnd % len(configs)]
    for c in order:
        for i, (cat, q) in enumerate(P):
            ids = tok.encode(chat(q), add_special_tokens=False)
            tps, out = configs[c](ids)
            res[c][i].append(tps)
            if rnd == 0 and c == "ours exact (k<=2)":
                if i not in refs:   # reference: stock (unfused) plain greedy
                    gdn(False); refs[i] = plain(target, ids, a.n)[0]
                same.append(out == refs[i][:len(out)] and len(out) == len(refs[i]))
        print(f"round {rnd} {c:<20} mean {statistics.mean(statistics.mean(v) for v in res[c]):.2f} tok/s  thermal {thermal_state()}", flush=True)

if a.llamacpp:
    runs = [("llama.cpp b10969", [])]
    if a.mtp_gguf and os.path.exists(a.mtp_gguf):
        runs.append(("llama.cpp b10969 + native MTP (n=3)", ["-md", a.mtp_gguf, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]))
    for label, extra in runs:
        p = subprocess.Popen([SERVER, "--model", os.path.expanduser(a.gguf), "--port", "8091", "--host", "127.0.0.1", "--no-webui", "--offline", "-ngl", "99",
                              "-np", "1", "-c", "4096", *extra], stdout=open(os.path.join(HERE, "logs/final_llama.log"), "w"), stderr=subprocess.STDOUT)
        while True:
            try: urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2); break
            except Exception: time.sleep(1)
        post("/completion", {"prompt": chat("hi"), "n_predict": 8})
        res[label] = [[post("/completion", {"prompt": chat(q), "n_predict": a.n, "temperature": 0, "cache_prompt": False})["timings"]["predicted_per_second"]
                       for _ in range(a.rounds)] for _, q in P]
        p.terminate(); p.wait()
        print(f"{label:<36} mean {statistics.mean(statistics.mean(v) for v in res[label]):.2f}", flush=True)


def boot_ci(xs, n=10000, seed=0):
    rng = random.Random(seed); ms = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


summary = {}
for c, per in res.items():
    pm = [statistics.mean(v) for v in per]
    lo, hi = boot_ci(pm)
    summary[c] = {"mean": round(statistics.mean(pm), 2), "ci95": [round(lo, 2), round(hi, 2)],
                  "by_category": {cat: round(statistics.mean(pm[i] for i, (cc, _) in enumerate(P) if cc == cat), 2) for cat in suite}}
base = summary["mlx-lm stock"]["mean"]
for c in summary:
    summary[c]["vs_mlx_lm"] = round(summary[c]["mean"] / base, 3)
summary["_exact_identical"] = f"{sum(same)}/{len(same)}"
summary["_thermal_end"] = thermal_state()
json.dump({"summary": summary, "per_prompt": res, "prompts": P}, open(os.path.join(HERE, f"results/final_{a.tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
