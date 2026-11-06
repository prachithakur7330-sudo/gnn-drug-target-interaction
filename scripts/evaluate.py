"""
Make the results plots from a finished training run:
  - predicted vs measured pKd scatter on the test set
  - training/validation loss curve

    python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dti.metrics import all_metrics  # noqa: E402

RES = Path("results")
FIG = Path("figures")


def main():
    FIG.mkdir(exist_ok=True)
    pred = np.load(RES / "test_predictions.npz")
    y, p = pred["y_true"], pred["y_pred"]
    metrics = all_metrics(y, p)
    print("test metrics:", json.dumps(metrics, indent=2))

    # scatter
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.hexbin(y, p, gridsize=45, cmap="viridis", bins="log", mincnt=1)
    lim = [min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2]
    ax.plot(lim, lim, "r--", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured pKd"); ax.set_ylabel("predicted pKd")
    ax.set_title(f"DAVIS test set  (MSE={metrics['mse']}, CI={metrics['ci']})")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_true.png", dpi=150); plt.close(fig)

    # loss curve
    history = json.loads((RES / "history.json").read_text())
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["train_mse"] for h in history], label="train MSE")
    ax.plot(ep, [h["val_mse"] for h in history], label="val MSE")
    best = min(history, key=lambda h: h["val_mse"])
    ax.axvline(best["epoch"], color="grey", ls=":", lw=1,
               label=f"best val (epoch {best['epoch']})")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_ylim(0, max(1.0, history[0]["val_mse"]))
    ax.legend(); ax.set_title("Training curve")
    fig.tight_layout(); fig.savefig(FIG / "training_curve.png", dpi=150); plt.close(fig)
    print(f"wrote {FIG}/pred_vs_true.png and {FIG}/training_curve.png")


if __name__ == "__main__":
    main()
