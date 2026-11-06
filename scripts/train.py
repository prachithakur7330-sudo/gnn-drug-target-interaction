"""
Train GraphDTA on DAVIS and report test metrics.

    python scripts/train.py --epochs 120

The DeepDTA test fold is held out for the final numbers. A tenth of the training pairs
is split off as a validation set for early stopping (split by drug-target pair, seeded
so it's reproducible). CPU-only is fine for DAVIS; it's a small dataset.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dti.data import DavisDataset  # noqa: E402
from dti.metrics import all_metrics  # noqa: E402
from dti.model import GraphDTA  # noqa: E402


def run_epoch(model, loader, proteins, optimiser, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        pred = model(batch, proteins)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimiser.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, proteins, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        ps.append(model(batch, proteins).cpu().numpy())
        ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="stop early and checkpoint; re-run with --resume to carry on")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = DavisDataset("data/processed", "train")
    test_set = DavisDataset("data/processed", "test")
    proteins = full_train.proteins.to(device)

    # carve a validation split out of the training pairs
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(full_train))
    n_val = len(full_train) // 10
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = full_train[list(train_idx)]
    val_set = full_train[list(val_idx)]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    model = GraphDTA().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  train={len(train_set)}  "
          f"val={len(val_set)}  test={len(test_set)}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ckpt_path = out / "checkpoint.pt"

    history = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    start_epoch = 1
    prior_seconds = 0.0

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        best_val = ckpt["best_val"]
        best_state = ckpt["best_state"]
        bad_epochs = ckpt["bad_epochs"]
        start_epoch = ckpt["epoch"] + 1
        prior_seconds = ckpt["seconds"]
        print(f"resumed from epoch {ckpt['epoch']} (best val {best_val:.4f})")

    start = time.time()
    stopped_early = False

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, proteins, optimiser, device)
        yv, pv = predict(model, val_loader, proteins, device)
        val_mse = float(np.mean((yv - pv) ** 2))
        scheduler.step(val_mse)
        history.append({"epoch": epoch, "train_mse": round(train_loss, 4), "val_mse": round(val_mse, 4)})

        if val_mse < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_mse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_mse:.4f}  best {best_val:.4f}",
              flush=True)

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history,
                    "best_val": best_val, "best_state": best_state, "bad_epochs": bad_epochs,
                    "epoch": epoch, "seconds": prior_seconds + time.time() - start}, ckpt_path)

        if bad_epochs >= args.patience:
            print(f"early stop at epoch {epoch} (no val improvement for {args.patience})")
            break
        if args.max_seconds and time.time() - start > args.max_seconds:
            print(f"time budget reached at epoch {epoch}; re-run with --resume to continue")
            stopped_early = True
            break

    model.load_state_dict(best_state)
    yt, pt = predict(model, test_loader, proteins, device)
    metrics = all_metrics(yt, pt)
    elapsed = round(prior_seconds + time.time() - start, 1)
    print("test metrics:", json.dumps(metrics))

    result = {
        "dataset": "DAVIS",
        "split": "DeepDTA fold setting 1",
        "n_params": n_params,
        "epochs_run": len(history),
        "completed": not stopped_early,
        "best_val_mse": round(best_val, 4),
        "train_seconds": elapsed,
        "test": metrics,
        "test_predictions_file": "test_predictions.npz",
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    np.savez(out / "test_predictions.npz", y_true=yt, y_pred=pt)
    torch.save(model.state_dict(), out / "graphdta_davis.pt")
    print(f"saved to {out}/  ({elapsed}s)")


if __name__ == "__main__":
    main()
