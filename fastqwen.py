"""Fast lossless greedy generation for Qwen3.5-9B on Apple Silicon (MLX + MTP self-speculation).

python fastqwen.py "your prompt" [-n 512] [--think]
Streams the answer and prints decode tok/s. Default mode: bit-identical output to plain greedy decoding of the MLX 4-bit model (fp16).
The stack is specmtp.py: MTP drafts with a trimmed-vocab head, verification through the multi-token kernel
(qmvt.py), DeltaNet state rollback, compiled capture layers, a fused DeltaNet kernel (fusedgdn.py) and adaptive draft length.
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
import fusedgdn, qmvt, specfast, specmtp
from specmlx import EOS

a = argparse.ArgumentParser()
a.add_argument("prompt"); a.add_argument("-n", type=int, default=512)
a.add_argument("--think", action="store_true", help="let the model think (default: thinking off)")
a.add_argument("--model", default=os.path.join(HERE, "../surgery/mlx/qwen35-9b-4bit"), help="MLX 4-bit Qwen3.5 dir")
a.add_argument("--mtp", default=None, help="MTP safetensors from fetch_mtp.py (default: models/mtp/mtp.safetensors)")
a.add_argument("--fast", action="store_true", help="allow 3-token drafts (4-token verify); a few % faster on easy text "
               "but can flip exact logit near-ties. Default mode is bit-identical to plain greedy decoding.")
a = a.parse_args()
qmvt.install(); specfast.install()
target, tok = load(a.model)
target.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
fusedgdn.install(target)   # one bit-exact Metal kernel per DeltaNet middle (+2-3%)
mtp = specmtp.load_mtp(target.language_model.args, mx.float16, a.mtp)
head = specmtp.Head(specmtp.head_module(target.language_model), specmtp.frequent_vocab(tok, 65536), specmtp.full_head_fn(target.language_model))
prompt = f"<|im_start|>user\n{a.prompt}<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if a.think else "<think>\n\n</think>\n\n")
ids = tok.encode(prompt, add_special_tokens=False)
gen, shown = [], 0
def show(new):
    global shown
    gen.extend(t for t in new if t not in EOS)
    text = tok.decode(gen)
    sys.stdout.write(text[shown:]); sys.stdout.flush(); shown = len(text)
specmtp.spec_mtp(target, mtp, head, tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False), 4, 1)  # warm up kernels
t0 = time.perf_counter()
from specpipe import spec_pipe
out, tps, acc, tpc = spec_pipe(target, mtp, head, ids, a.n, 0, 3 if a.fast else 2, on_tokens=show)
print(f"\n\n[{len(out)} tokens, {tps:.1f} tok/s decode, draft acceptance {acc:.0%}, {tpc:.2f} tokens/step]")
