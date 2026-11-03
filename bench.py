"""Thermal-aware decode-speed benchmark for qwen3.5:9b on this Mac.

python bench.py NAME [--engine llamacpp|mlx] [--model PATH] [--cool 60] [-- extra llama-server flags...]

For each config: wait until macOS thermal state is "nominal" and the GPU has idled >= --cool s, start the engine,
run 1 warmup + 4 fixed prompts (2 math, code, prose; thinking off, greedy, 256 new tokens) + 1 long-prompt prefill.
Reports median decode tok/s (engine-reported), prefill tok/s, and the thermal state before/after. Saves outputs to
results/NAME.json so lossless tricks can be checked for byte-identical text against the baseline.
"""
import argparse, ctypes, ctypes.util, json, os, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.environ.get("LLAMA_SERVER", "/Applications/Ollama.app/Contents/Resources/llama-server")   # override to test another build
GGUF = os.path.realpath(os.path.join(HERE, "../surgery/models/base.gguf"))
MLX = os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit")
BOX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PROMPTS = {
    "math_gsm": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?" + BOX,
    "math_alg": "Find all real numbers $x$ such that $x^4 - 5x^2 + 4 = 0$, and compute the sum of their squares." + BOX,
    "code": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Include a docstring, type hints, and three pytest test cases.",
    "prose": "Explain how a household refrigerator moves heat from inside to outside. Cover the compressor, condenser, expansion valve and evaporator.",
}
LONG = "Summarize the following text in one sentence.\n\n" + " ".join(
    f"Section {i}: the committee reviewed budget line {i}, noted variance of {i*3 % 17} percent, and deferred the decision." for i in range(60))
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def thermal_state():
    """NSProcessInfo.thermalState via the ObjC runtime: 0 nominal, 1 fair, 2 serious, 3 critical."""
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]
    send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    send_l = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    pi = send(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    return ["nominal", "fair", "serious", "critical"][send_l(pi, objc.sel_registerName(b"thermalState"))]


def cooldown(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds or thermal_state() != "nominal":
        time.sleep(5)
    return round(time.time() - t0)


def post(path, body, port=8091):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def run_llamacpp(model, flags):
    log = open(os.path.join(HERE, "logs/server.log"), "w")
    p = subprocess.Popen([SERVER, "--model", model, "--port", "8091", "--host", "127.0.0.1", "--no-webui", "--offline",
                          "-ngl", "99", "-np", "1", "-c", "4096", *flags], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2); break
            except Exception:
                if p.poll() is not None: raise RuntimeError("server died; see logs/server.log")
                time.sleep(1)
        gen = lambda q, n: post("/completion", {"prompt": chat(q), "n_predict": n, "temperature": 0, "seed": 0, "cache_prompt": False})
        gen("Say hi.", 16)  # warmup: Metal pipeline compile, page-in
        out = {}
        for k, q in PROMPTS.items():
            r = gen(q, 256); t = r["timings"]
            out[k] = {"tps": t["predicted_per_second"], "n": t["predicted_n"], "text": r["content"],
                      "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}
        r = gen(LONG, 1)
        return out, r["timings"]["prompt_per_second"], r["timings"]["prompt_n"]
    finally:
        p.terminate(); p.wait()


def run_mlx(model, flags):
    from mlx_lm import load, stream_generate
    m, tok = load(model)
    if "--fp16" in flags:  # M1 has no native bf16: cast scales/biases/norms/etc to fp16 (packed 4-bit ints untouched)
        import mlx.core as mx
        from mlx.utils import tree_map
        m.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, m.parameters()))
    if "--compile" in flags:  # fuse each DeltaNet layer's single-token step (fastdecode.py)
        sys.path.insert(0, HERE); import fastdecode
        fastdecode.install()
    def gen(q, n):
        last, text = None, ""
        for resp in stream_generate(m, tok, chat(q), max_tokens=n):
            last, text = resp, text + resp.text
        return last, text
    gen("Say hi.", 16)
    out = {}
    for k, q in PROMPTS.items():
        r, text = gen(q, 256)
        out[k] = {"tps": r.generation_tps, "n": r.generation_tokens, "text": text}
    r, _ = gen(LONG, 1)
    return out, r.prompt_tps, r.prompt_tokens


def main():
    a = argparse.ArgumentParser()
    a.add_argument("name"); a.add_argument("--engine", default="llamacpp"); a.add_argument("--model")
    a.add_argument("--cool", type=int, default=60)
    argv = sys.argv[1:]
    own, flags = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    a = a.parse_args(own)
    waited, before = cooldown(a.cool), thermal_state()
    t0 = time.time()
    out, pp, pn = (run_mlx if a.engine == "mlx" else run_llamacpp)(a.model or (MLX if a.engine == "mlx" else GGUF), flags)
    res = {"name": a.name, "engine": a.engine, "flags": flags, "decode_tps_median": statistics.median(v["tps"] for v in out.values()),
           "per_prompt_tps": {k: round(v["tps"], 2) for k, v in out.items()}, "prefill_tps": pp, "prefill_tokens": pn,
           "thermal_before": before, "thermal_after": thermal_state(), "cooldown_s": waited, "load_s": round(time.time() - t0),
           "outputs": out}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, f"results/{a.name}.json"), "w"), indent=1)
    base = os.path.join(HERE, "results/baseline.json")
    same = ""
    if a.name != "baseline" and os.path.exists(base):
        b = json.load(open(base))["outputs"]
        same = f"  identical-to-baseline {sum(b[k]['text'] == out[k]['text'] for k in out)}/{len(out)}"
    print(f"{a.name:<28} decode {res['decode_tps_median']:6.2f} tok/s  {res['per_prompt_tps']}  prefill {pp:7.1f} tok/s  "
          f"thermal {before}->{res['thermal_after']}  load {res['load_s']}s{same}")


if __name__ == "__main__":
    main()
