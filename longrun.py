"""Sustained test: 1500-token generations with thinking ON (long context growth, thermal soak).
Compares: llama.cpp (Ollama engine, Q4_K_M), MLX plain greedy (fp16), MLX optimized (MTP + pipelined spec)."""
import json, os, statistics, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
import qmvt, specfast, specmtp
from specmlx import EOS, plain
from specpipe import spec_pipe
from bench import SERVER, GGUF, post, thermal_state
N = 1500
QS = ["A train leaves at 3pm going 60 mph; another leaves the same station at 4pm going 80 mph in the same direction. When and where does the second catch up? Explain.",
      "Write a Python class implementing an LRU cache with O(1) get and put, with tests.",
      "Compare the causes of World War I and World War II in a structured essay."]
think = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
res = {}
# 1) llama.cpp
p = subprocess.Popen([SERVER, "--model", GGUF, "--port", "8091", "--host", "127.0.0.1", "--no-webui", "--offline", "-ngl", "99",
                      "-np", "1", "-c", "8192"], stdout=open(os.path.join(HERE, "logs/longrun_server.log"), "w"), stderr=subprocess.STDOUT)
while True:
    try: urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2); break
    except Exception: time.sleep(1)
post("/completion", {"prompt": think("hi"), "n_predict": 8})
res["llama.cpp"] = [post("/completion", {"prompt": think(q), "n_predict": N, "temperature": 0, "cache_prompt": False})["timings"]["predicted_per_second"] for q in QS]
p.terminate(); p.wait()
print("llama.cpp", [round(x, 1) for x in res["llama.cpp"]], thermal_state(), flush=True)
# 2) MLX plain and 3) MLX optimized
qmvt.install(); specfast.install()
target, tok = load(os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"))
target.update(tree_map(lambda q: q.astype(mx.float16) if q.dtype == mx.bfloat16 else q, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
mtp = specmtp.load_mtp(target.language_model.args, mx.float16)
head = specmtp.Head(target.language_model.lm_head, specmtp.frequent_vocab(tok, 65536))
warm = tok.encode(think("hi"), add_special_tokens=False); plain(target, warm, 8); spec_pipe(target, mtp, head, warm, 8, 1)
res["mlx plain"], res["mlx optimized"], same = [], [], []
for q in QS:
    ids = tok.encode(think(q), add_special_tokens=False)
    ref, t0 = plain(target, ids, N); res["mlx plain"].append(t0)
    got, t1, acc, tpc = spec_pipe(target, mtp, head, ids, N, 0, 3); res["mlx optimized"].append(t1)
    same.append(f"{next((j for j, (x, z) in enumerate(zip(ref, got)) if x != z), min(len(ref), len(got)))}/{len(ref)}")
    print(f"plain {t0:.1f}  optimized {t1:.1f} (accept {acc:.0%}, {tpc:.2f} tok/step, identical prefix {same[-1]})", thermal_state(), flush=True)
print(json.dumps({k: [round(x, 1) for x in v] for k, v in res.items()}))
print("means:", {k: round(statistics.mean(v), 1) for k, v in res.items()})
