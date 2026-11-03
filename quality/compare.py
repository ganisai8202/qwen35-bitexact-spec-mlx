"""compare.py N SET [TAG ...]  -> accuracy of each tag vs 'base' on the first N problems of SET, paired.
KEY=ok2048 env scores at the 2048-token cap instead of the extended 4096 cap. fixed = base wrong -> tag right, broke = base right -> tag wrong, p = exact two-sided McNemar test."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import comb
from evalmath import load
KEY = os.environ.get("KEY", "ok")
ok = lambda r: r.get(KEY, r["ok"])

def res(tag, s):
    f = f"results/{tag}/{s}.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if os.path.exists(f) else {}

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)

n, sets = int(sys.argv[1]), sys.argv[2].split(",")
tags = sys.argv[3:] or sorted(t for t in os.listdir("results") if not t.startswith("smoke"))
print(f"{'tag':<16}" + "".join(f"{s+' acc':>14}{'fix/brk':>9}{'p':>7}" for s in sets) + f"{'combined':>10}{'fix/brk':>9}{'p':>7}{'toks':>6}")
for t in tags:
    row, F, B, OK, NN, toks = f"{t:<16}", 0, 0, 0, 0, []
    for s in sets:
        a, b = res("base", s), res(t, s)
        try:
            ids = [p[0] for p in load(s)[:n]]
        except FileNotFoundError:   # datasets not shipped: exact-mode files hold the first n seeded problems, in order
            ids = list(b)[:n]
        ids = [i for i in ids if i in a and i in b]
        if not ids:
            row += f"{'-':>14}{'':>9}{'':>7}"; continue
        fx = sum(not ok(a[i]) and ok(b[i]) for i in ids); br = sum(ok(a[i]) and not ok(b[i]) for i in ids)
        nok = sum(ok(b[i]) for i in ids); F += fx; B += br; OK += nok; NN += len(ids); toks += [b[i]["toks"] for i in ids]
        row += f"{nok/len(ids):>9.3f}({len(ids):>3})" + f"{f'+{fx}/-{br}':>9}{mcnemar(fx, br):>7.2f}"
    if NN:
        row += f"{OK/NN:>10.3f}{f'+{F}/-{B}':>9}{mcnemar(F, B):>7.2f}{sum(toks)/len(toks):>6.0f}"
    print(row)
