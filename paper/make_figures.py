"""Paper figures from results/*.json|csv (skips any figure whose inputs are missing)."""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(F, exist_ok=True)

# 1) verification cost vs T (kernel_compare.csv)
p = os.path.join(R, "kernel_compare.csv")
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    mats = sorted({r["matrix"] for r in rows})
    fig, axs = plt.subplots(2, (len(mats) + 1) // 2, figsize=(6.4, 4.6), sharey=True)
    axs = axs.ravel()
    for ax, m in zip(axs, mats):
        rr = [r for r in rows if r["matrix"] == m]
        T = [1] + [int(r["T"]) for r in rr]
        for key, lab, st in (("mlx_dispatch_x", "MLX (dispatched)", "o-"), ("mlx_qmv_wide_x", "MLX qmv_wide (port)", "s--"), ("qmvt_x", "ours", "^-")):
            ax.plot(T, [1.0] + [float(r[key]) for r in rr], st, label=lab, ms=4)
        ax.plot(T, T, ":", color="gray", lw=0.8, label="T x (no reuse)")
        ax.set_title(m, fontsize=8); ax.set_xlabel("tokens verified T", fontsize=8); ax.set_xticks(T)
    axs[0].set_ylabel("cost / single-token kernel"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "tscale.pdf")); print("tscale.pdf")

# 2) ablation waterfall (numbers from the paper's ablation table)
steps = [("llama.cpp b10969", 22.4), ("stock MLX", 34.8), ("+MTP spec", 40.2), ("+verify kernel", 43.8), ("+trimmed head", 46.6),
         ("+43.8k head, adaptive k", 49.1), ("+compiled capture", 50.2), ("+tuned registers", 52.2), ("+pipelining", 55.6)]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.bar(range(len(steps)), [v for _, v in steps], color=["#999"] * 2 + ["#4a7"] * (len(steps) - 2))
ax.set_xticks(range(len(steps))); ax.set_xticklabels([s for s, _ in steps], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("decode tok/s"); ax.axhline(42, ls=":", color="gray", lw=0.8); ax.text(0, 42.8, "plain-decoding bandwidth bound (~42)", fontsize=6, color="gray")
for i, (_, v) in enumerate(steps): ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(F, "ablation.pdf")); print("ablation.pdf")

# 3) per-problem speed distribution, exact mode on GSM8K/MATH (quality_spec logs)
qs = [os.path.join(R, "../../surgery/results/mlx_spec", f) for f in ("gsm_dev.jsonl", "math_dev.jsonl")]
data = {os.path.basename(q).split("_")[0].upper(): [json.loads(l)["tps"] for l in open(q)] for q in qs if os.path.exists(q)}
if data:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k, v in data.items(): ax.hist(v, bins=20, alpha=0.6, label=f"{k} (n={len(v)})")
    ax.axvline(34.8, ls=":", color="gray"); ax.text(35.2, ax.get_ylim()[1] * 0.9, "stock MLX", fontsize=6, color="gray")
    ax.set_xlabel("decode tok/s per problem (exact mode)"); ax.set_ylabel("problems"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "per_problem.pdf")); print("per_problem.pdf")
