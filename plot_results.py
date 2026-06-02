"""
plot_results.py
===============
Load training_results.json and generate publication-quality training curves.

Usage
-----
    python plot_results.py                        # reads training_results.json
    python plot_results.py --file my_results.json

Produces two plots:
    training_curves.png  — scores + 20-ep moving average + loss curve
"""

import json
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib not found — install with:  pip install matplotlib")
    raise


def plot(results_file: str = "training_results.json"):
    with open(results_file) as f:
        data = json.load(f)

    scores     = data["scores"]
    losses     = data["losses"]
    moving_avg = data["moving_avg"]
    episodes   = list(range(1, len(scores) + 1))

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # ── Left: Score curve ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(episodes, scores, alpha=0.15, color="#4A9EE8", label="Episode score")
    ax1.plot(episodes, scores,     alpha=0.3,  color="#4A9EE8", linewidth=0.7)
    ax1.plot(episodes, moving_avg, color="#E84A4A", linewidth=2.0, label="20-ep average")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score (food eaten)")
    ax1.set_title("Training Progress — Score")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate best score
    best_ep    = int(np.argmax(scores)) + 1
    best_score = max(scores)
    ax1.annotate(
        f"Best: {best_score}",
        xy=(best_ep, best_score),
        xytext=(best_ep + len(scores) * 0.05, best_score * 0.95),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray",
    )

    # ── Right: Loss curve ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    # Smooth the loss for readability
    smooth_loss = []
    win = 20
    for i in range(len(losses)):
        smooth_loss.append(np.mean(losses[max(0, i - win):i + 1]))

    ax2.plot(episodes, losses,      alpha=0.2,  color="#9B4AE8", linewidth=0.7, label="Raw loss")
    ax2.plot(episodes, smooth_loss, color="#9B4AE8", linewidth=2.0, label=f"{win}-ep smooth")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Huber loss")
    ax2.set_title("Training Loss")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("DQN Snake Agent — Training Curves", fontsize=13, fontweight="bold", y=1.02)

    out = "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="training_results.json")
    args = p.parse_args()
    plot(args.file)
