"""Analyze results.jsonl: best run per variant, loss-vs-tokens curves,
tokens-to-reach-baseline-loss ratios, and a summary plot."""
import json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
curves, finals = defaultdict(list), {}
for line in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(line)
    key = (r["run"], r["lr"])
    if "final_val_loss" in r:
        finals[key] = r["final_val_loss"]
    else:
        curves[key].append((r["tokens"], r["val_loss"]))

# best lr per variant by final val loss
best = {}
for (run, lr), v in finals.items():
    if run not in best or v < finals[(run, best[run])]:
        best[run] = lr

ORDER = ["gpt2-adamw", "modern-adamw", "modern-muon", "shortcut-muon"]
present = [r for r in ORDER if r in best]
print(f"{'variant':<15} {'best lr':>8} {'final val loss':>15}  (all lrs tried)")
for run in present:
    tried = {lr: finals[(run, lr)] for (r, lr) in finals if r == run}
    print(f"{run:<15} {best[run]:>8} {finals[(run, best[run])]:>15.4f}  {tried}")

# tokens needed to reach the baseline's final loss
if "gpt2-adamw" in best:
    target = finals[("gpt2-adamw", best["gpt2-adamw"])]
    base_budget = max(t for t, _ in curves[("gpt2-adamw", best["gpt2-adamw"])])
    print(f"\ntokens to reach baseline final val loss ({target:.4f}):")
    for run in present:
        cur = sorted(curves[(run, best[run])])
        hit = next((t for t, v in cur if v <= target), None)
        if hit:
            print(f"  {run:<15} {hit:>10,} tokens  ({base_budget / hit:.1f}x fewer than baseline)")
        else:
            print(f"  {run:<15} never reached within budget")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    colors = {"gpt2-adamw": "#888888", "modern-adamw": "#4477aa",
              "modern-muon": "#ee6677", "shortcut-muon": "#228833"}
    for run in present:
        cur = sorted(curves[(run, best[run])])
        ax.plot([t / 1e6 for t, _ in cur], [v for _, v in cur],
                marker="o", ms=3, label=f"{run} (lr={best[run]})", color=colors.get(run))
    if "gpt2-adamw" in best:
        ax.axhline(finals[("gpt2-adamw", best["gpt2-adamw"])], ls=":", c="#888888", lw=1)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("val loss (nats/byte)")
    ax.set_title("CPU micro-scale sample efficiency (1.25M-param byte-level GPT)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "curves.png"))
    print("\nwrote curves.png")
except Exception as e:
    print("plot skipped:", e)
