"""Cost of processing T new tokens in one llama.cpp forward (what speculative verification pays).
Caches a base prompt, then repeatedly sends base + T fresh tokens with n_predict=1; reads the server's
prompt-eval time for those T tokens. Median of 7. Uses Ollama's bundled llama-server (b10969)."""
import json, os, statistics, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench import SERVER, GGUF, chat, cooldown, post

flags = sys.argv[1:]
log = open(os.path.join(HERE, "logs/tscale.log"), "w")
p = subprocess.Popen([SERVER, "--model", GGUF, "--port", "8091", "--host", "127.0.0.1", "--no-webui", "--offline",
                      "-ngl", "99", "-np", "1", "-c", "4096", *flags], stdout=log, stderr=subprocess.STDOUT)
try:
    while True:
        try: urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2); break
        except Exception: time.sleep(1)
    cooldown(20)
    base = chat("Explain photosynthesis briefly.")
    post("/completion", {"prompt": base, "n_predict": 1, "cache_prompt": True})
    words = " apple banana cherry delta echo fox golf hotel india juliet kilo lima mike".split()
    res = {}
    for T in (1, 2, 3, 4, 5, 6, 8):
        ts = []
        for r in range(7):
            extra = "".join(words[(r + j) % len(words)] for j in range(T))  # ~1 token per word
            t = post("/completion", {"prompt": base + extra, "n_predict": 1, "cache_prompt": True, "temperature": 0})["timings"]
            if t["prompt_n"] == T:
                ts.append(t["prompt_ms"])
        res[T] = round(statistics.median(ts), 2) if ts else None
    print("llama.cpp T-token forward ms:", res, " ratio vs T=1:", {T: round(v / res[1], 2) for T, v in res.items() if v})
finally:
    p.terminate(); p.wait()
