"""Smoke test for another Qwen3.5 size: MTP teacher-forced next-next accuracy + exact-mode identity (64 tokens).
python smoke_model.py MODEL_DIR MTP_PATH  -> exit 1 if MTP accuracy < 0.4 or exact mode differs from plain greedy."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
import qmvt, specfast, specmtp
from specmlx import EOS, plain
from specpipe import spec_pipe
from bench import chat, PROMPTS
qmvt.install(); specfast.install()
target, tok = load(sys.argv[1])
target.update(tree_map(lambda q: q.astype(mx.float16) if q.dtype == mx.bfloat16 else q, target.parameters()))
EOS.update({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
lm = target.language_model
mtp = specmtp.load_mtp(lm.args, mx.float16, sys.argv[2])
head = specmtp.Head(specmtp.head_module(lm), specmtp.frequent_vocab(tok, 65536), specmtp.full_head_fn(lm))
r = json.loads(open(os.path.join(HERE, "../surgery/results/base/math_dev.jsonl")).readline())
ids = tok.encode(chat("Solve the problem.") + r["text"], add_special_tokens=False)[:400]
x = mx.array([ids]); h = lm.model(x, cache=make_prompt_cache(target)); E = lm.model.embed_tokens(x)
pred = mx.argmax(specmtp.full_head_fn(lm)(mtp(h[:, :-2], E[:, 1:-1], KVCache())), axis=-1)[0]
acc = (pred == x[0, 2:]).astype(mx.float32).mean().item()
ok_all = True
for q in list(PROMPTS.values())[:2]:
    pids = tok.encode(chat(q), add_special_tokens=False)
    ref = plain(target, pids, 64)[0]; got = spec_pipe(target, mtp, head, pids, 64, 0, 2)[0]
    ok_all &= got == ref
print(f"SMOKE {os.path.basename(sys.argv[1])}: MTP teacher-forced acc {acc:.3f}, exact-mode identical {ok_all}")
sys.exit(0 if acc >= 0.4 and ok_all else 1)
