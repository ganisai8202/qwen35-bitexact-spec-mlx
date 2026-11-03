"""Fetch ONLY the 15 mtp.* tensors of Qwen/Qwen3.5-9B (~0.5 GB of an ~18 GB checkpoint) via HTTP range requests.
Each safetensors shard starts with an 8-byte header length + a JSON header giving every tensor's byte range.
Writes models/mtp/mtp.safetensors (bf16, original names). Raw bytes only; nothing downloaded is executed."""
import json, os, struct, urllib.request
import numpy as np
import mlx.core as mx

import sys
REPO = f"https://huggingface.co/{sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen3.5-9B'}/resolve/main/"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "models/mtp/mtp.safetensors")


def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    assert len(data) == end - start, (url, start, end, len(data))
    return data


index = json.load(urllib.request.urlopen(REPO + "model.safetensors.index.json"))["weight_map"]
want = {k: v for k, v in index.items() if k.startswith("mtp.")}
tensors = {}
for shard in sorted(set(want.values())):
    url = REPO + shard
    hlen = struct.unpack("<Q", get(url, 0, 8))[0]
    header = json.loads(get(url, 8, 8 + hlen))
    base = 8 + hlen
    for name in sorted(k for k, s in want.items() if s == shard):
        meta = header[name]
        assert meta["dtype"] == "BF16", meta
        s, e = meta["data_offsets"]
        raw = np.frombuffer(get(url, base + s, base + e), dtype=np.uint16).reshape(meta["shape"])
        tensors[name] = mx.array(raw).view(mx.bfloat16)
        print(f"{name:<55} {meta['shape']}  {(e - s) / 1e6:7.1f} MB", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
mx.save_safetensors(OUT, tensors)
print("saved", OUT, f"{os.path.getsize(OUT) / 1e6:.0f} MB")
