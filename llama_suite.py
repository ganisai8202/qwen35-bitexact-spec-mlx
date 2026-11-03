"""llama.cpp (Ollama's bundled llama-server, or $LLAMA_SERVER) on the 20-prompt suite: same prompts, 256 tokens,
greedy, ROUNDS rounds, plain and native-MTP (n=3). No MLX model is loaded.  -> results/llama_suite_<tag>.json
python llama_suite.py TAG [--gguf PATH] [--mtp-gguf PATH|""]"""
import argparse, json, os, random, statistics, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench import SERVER, GGUF, post, thermal_state

a = argparse.ArgumentParser(); a.add_argument("tag"); a.add_argument("--gguf", default=GGUF); a.add_argument("--rounds", type=int, default=3)
a.add_argument("--mtp-gguf", default=os.path.join(HERE, "models/mtp_gguf/Qwen3.5-9B-MTP-ONLY-Q4_K_M.gguf")); a = a.parse_args()
suite = json.load(open(os.path.join(HERE, "prompts_suite.json")))
P = [(cat, q) for cat, qs in suite.items() for q in qs]
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
runs = [("llama.cpp", [])] + ([("llama.cpp + native MTP (n=3)", ["-md", a.mtp_gguf, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])]
                              if a.mtp_gguf and os.path.exists(a.mtp_gguf) else [])
res = {}
for label, extra in runs:
    p = subprocess.Popen([SERVER, "--model", os.path.expanduser(a.gguf), "--port", "8091", "--host", "127.0.0.1", "--no-webui", "--offline",
                          "-ngl", "99", "-np", "1", "-c", "4096", *extra], stdout=open(os.path.join(HERE, "logs/llama_suite.log"), "w"), stderr=subprocess.STDOUT)
    while True:
        try: urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2); break
        except Exception:
            if p.poll() is not None: sys.exit("server died; see logs/llama_suite.log")
            time.sleep(1)
    post("/completion", {"prompt": chat("hi"), "n_predict": 8})
    res[label] = [[post("/completion", {"prompt": chat(q), "n_predict": 256, "temperature": 0, "cache_prompt": False})["timings"]["predicted_per_second"]
                   for _ in range(a.rounds)] for _, q in P]
    p.terminate(); p.wait()
rng, summary = random.Random(0), {}
for c, per in res.items():
    pm = [statistics.mean(v) for v in per]
    ms = sorted(statistics.mean(rng.choices(pm, k=len(pm))) for _ in range(10000))
    summary[c] = {"mean": round(statistics.mean(pm), 2), "ci95": [round(ms[250], 2), round(ms[9750], 2)],
                  "by_category": {cat: round(statistics.mean(pm[i] for i, (cc, _) in enumerate(P) if cc == cat), 2) for cat in suite}}
    print("LLAMA", c, json.dumps(summary[c]), flush=True)
summary["_thermal_end"] = thermal_state()
json.dump({"summary": summary, "per_prompt": res, "server": SERVER}, open(os.path.join(HERE, f"results/llama_suite_{a.tag}.json"), "w"), indent=1)
