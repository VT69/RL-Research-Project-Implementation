"""
train.py
========
Training and evaluation loop for GLD²-GNN (paper Section IV).

Implements
----------
  • Cross-dataset validation  (e.g. Ga+Ju → Si, paper Table IV)
  • Mixed-data k-fold cross-validation  (5-fold / 10-fold, paper Table V)
  • Early stopping with patience=30  (paper Section IV-A-2)
  • Adam optimiser: β1=0.9, β2=0.995, weight_decay=5e-4
  • CosineAnnealingLR: T_max=14 epochs, η_min=1e-5
  • Binary cross-entropy loss
  • Metrics: Accuracy, F1 score, Geometric Mean  (equations 20-21)
  • Vote-based subject-level prediction for confusion matrices

Quick start
-----------
  # Cross-dataset:  Ga+Ju → Si
  python train.py \
      --data_root ./data \
      --mode      cross \
      --train_sets Ga Ju \
      --test_set   Si \
      --T          128 \
      --epochs     120 \
      --batch_size 64

  # 5-fold cross-validation on all datasets
  python train.py \
      --data_root  ./data \
      --mode       kfold \
      --k_folds    5 \
      --T          128 \
      --epochs     120 \
      --batch_size 64
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing  import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from data_loader import build_dataloaders, build_kfold_datasets
from model       import GLD2GNN


# ---------------------------------------------------------------------------
# Metrics  (equations 20 & 21)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes Accuracy, F1, and Geometric Mean.

    Geometric Mean = sqrt(Precision * Recall)  — equation 21.
    All three metrics are reported in the paper (Tables IV & V).

    Args:
        y_true : ground-truth labels  [N]  int
        y_pred : predicted labels     [N]  int

    Returns:
        dict with keys 'acc', 'f1', 'gmean'
    """
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # Gmean from confusion matrix to handle edge cases cleanly
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    gmean     = float(np.sqrt(precision * recall))

    return {
        "acc":   float(acc),
        "f1":    float(f1),
        "gmean": gmean,
    }


# ---------------------------------------------------------------------------
# Vote-based subject-level prediction
# ---------------------------------------------------------------------------

def vote_predict(
    model:     GLD2GNN,
    loader:    DataLoader,
    device:    torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference on all gait-cycle samples and returns sample-level
    predictions (for metric computation) and probabilities.

    For subject-level voting (paper Section IV-C), aggregate the returned
    probs by subject ID externally using majority vote.

    Returns:
        y_true : [N]  ground-truth labels
        y_pred : [N]  predicted labels
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            ts_nodes = batch["ts_nodes"].to(device)
            ts_edges = batch["ts_edges"].to(device)
            ms_nodes = batch["ms_nodes"].to(device)
            ms_edges = batch["ms_edges"].to(device)
            labels   = batch["label"].to(device)

            probs  = model(ts_nodes, ts_edges, ms_nodes, ms_edges)  # [B,1]
            preds  = (probs.squeeze(1) >= threshold).long()

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    return np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model:     GLD2GNN,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """
    Runs one full pass over the training set.

    Returns:
        mean training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        ts_nodes = batch["ts_nodes"].to(device)
        ts_edges = batch["ts_edges"].to(device)
        ms_nodes = batch["ms_nodes"].to(device)
        ms_edges = batch["ms_edges"].to(device)
        labels   = batch["label"].float().to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                probs = model(ts_nodes, ts_edges, ms_nodes, ms_edges)
                loss  = criterion(probs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            probs = model(ts_nodes, ts_edges, ms_nodes, ms_edges)
            loss  = criterion(probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    model:        GLD2GNN,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 120,
    lr:           float = 5e-4,
    weight_decay: float = 5e-4,
    patience:     int   = 30,
    use_amp:      bool  = False,
    save_dir:     Optional[str] = None,
    run_name:     str   = "run",
    verbose:      bool  = True,
) -> Dict:
    """
    Full training loop with early stopping and cosine LR annealing.

    Optimiser : Adam  β1=0.9, β2=0.995, weight_decay=5e-4
    Scheduler : CosineAnnealingLR  T_max=14, η_min=1e-5
    Loss      : Binary cross-entropy
    Early stop: patience=30 epochs on validation loss

    Args:
        model        : GLD2GNN instance (already on device)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        test_loader  : test DataLoader
        device       : torch device
        epochs       : maximum training epochs (paper uses 120)
        lr           : initial learning rate (paper: 5e-4)
        weight_decay : L2 regularisation (paper: 5e-4)
        patience     : early-stopping patience (paper: 30)
        use_amp      : mixed-precision training (recommended on RTX GPUs)
        save_dir     : directory to save best model checkpoint
        run_name     : prefix for saved files
        verbose      : print per-epoch metrics

    Returns:
        results dict with train/val/test metrics and training history
    """
    model = model.to(device)

    # Optimiser — paper Section IV-A-2
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.995),
        weight_decay=weight_decay,
    )

    # Scheduler — CosineAnnealingLR, T_max=14, η_min=1e-5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=14, eta_min=1e-5
    )

    criterion = nn.BCELoss()
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_path = os.path.join(save_dir, f"{run_name}_best.pt")
    else:
        best_path = None

    # Early stopping state
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0
    best_epoch     = 0

    # History
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [], "val_gmean": [],
    }

    if verbose:
        header = (f"{'Ep':>4}  {'LR':>8}  {'TrLoss':>8}  "
                  f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaF1':>7}  "
                  f"{'VaGm':>7}  {'α':>6}")
        print(header)
        print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training
        tr_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validation
        y_true_v, y_pred_v = vote_predict(model, val_loader, device)
        val_metrics = compute_metrics(y_true_v, y_pred_v)

        # Validation loss (for early stopping)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                probs = model(
                    batch["ts_nodes"].to(device),
                    batch["ts_edges"].to(device),
                    batch["ms_nodes"].to(device),
                    batch["ms_edges"].to(device),
                )
                lbl = batch["label"].float().to(device).unsqueeze(1)
                val_loss += criterion(probs, lbl).item() * lbl.size(0)
        val_loss /= len(val_loader.dataset)

        # Scheduler step
        scheduler.step()

        # Adaptive matrix warm-up step (once per epoch)
        model.step()

        # History
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_gmean"].append(val_metrics["gmean"])

        # Early stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_metrics["acc"]
            patience_count = 0
            best_epoch     = epoch
            if best_path:
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_loss":   val_loss,
                    "val_acc":    val_metrics["acc"],
                    "optimizer":  optimizer.state_dict(),
                }, best_path)
        else:
            patience_count += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            cur_lr = scheduler.get_last_lr()[0]
            alpha  = model.alpha.item()
            print(
                f"{epoch:>4}  {cur_lr:>8.6f}  {tr_loss:>8.4f}  "
                f"{val_loss:>8.4f}  {val_metrics['acc']:>7.4f}  "
                f"{val_metrics['f1']:>7.4f}  {val_metrics['gmean']:>7.4f}  "
                f"{alpha:>6.4f}  "
                f"{'*' if patience_count == 0 else ''}"
            )

        if patience_count >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best epoch={best_epoch})")
            break

    # Load best checkpoint for test evaluation
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if verbose:
            print(f"\n  Loaded best checkpoint (epoch {ckpt['epoch']}, "
                  f"val_acc={ckpt['val_acc']:.4f})")

    # Final test evaluation
    y_true_t, y_pred_t = vote_predict(model, test_loader, device)
    test_metrics = compute_metrics(y_true_t, y_pred_t)

    if verbose:
        cm = confusion_matrix(y_true_t, y_pred_t, labels=[0, 1])
        print(f"\n  Test results:")
        print(f"    Acc  = {test_metrics['acc']:.4f}")
        print(f"    F1   = {test_metrics['f1']:.4f}")
        print(f"    Gmean= {test_metrics['gmean']:.4f}")
        print(f"    Confusion matrix:\n      {cm}")

    return {
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc":  best_val_acc,
        "test_acc":      test_metrics["acc"],
        "test_f1":       test_metrics["f1"],
        "test_gmean":    test_metrics["gmean"],
        "history":       history,
        "y_true":        y_true_t.tolist(),
        "y_pred":        y_pred_t.tolist(),
    }


# ---------------------------------------------------------------------------
# Cross-dataset validation runner
# ---------------------------------------------------------------------------

def run_cross_dataset(args) -> None:
    """
    Runs cross-dataset validation once (e.g. Ga+Ju → Si).
    Experiment is repeated `args.repeats` times (paper repeats 4×)
    and results are averaged.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\n  Device : {device}")
    print(f"  Train  : {args.train_sets}  →  Test: {args.test_set}")

    all_results = []

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        print(f"\n{'='*60}")
        print(f"  Repeat {repeat+1}/{args.repeats}  (seed={seed})")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader = build_dataloaders(
            data_root   = args.data_root,
            train_sets  = args.train_sets,
            test_set    = args.test_set,
            T           = args.T,
            aug_factor  = args.aug_factor,
            batch_size  = args.batch_size,
            val_split   = 0.2,
            seed        = seed,
        )

        model = GLD2GNN(T=args.T, device=device).to(device)

        run_name = (f"cross_{''.join(args.train_sets)}_"
                    f"{args.test_set}_r{repeat}")

        result = train(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
            epochs       = args.epochs,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            patience     = args.patience,
            use_amp      = args.amp,
            save_dir     = args.save_dir,
            run_name     = run_name,
            verbose      = True,
        )
        all_results.append(result)

    # Aggregate across repeats
    accs   = [r["test_acc"]   for r in all_results]
    f1s    = [r["test_f1"]    for r in all_results]
    gmeans = [r["test_gmean"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"  Cross-dataset results  "
          f"({'+'}.join(args.train_sets)) → {args.test_set}")
    print(f"{'='*60}")
    print(f"  Acc  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
    print(f"  F1   : {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")
    print(f"  Gmean: {np.mean(gmeans)*100:.2f} ± {np.std(gmeans)*100:.2f}")

    if args.save_dir:
        out = {
            "mode":       "cross_dataset",
            "train_sets": args.train_sets,
            "test_set":   args.test_set,
            "acc_mean":   float(np.mean(accs)),
            "acc_std":    float(np.std(accs)),
            "f1_mean":    float(np.mean(f1s)),
            "f1_std":     float(np.std(f1s)),
            "gmean_mean": float(np.mean(gmeans)),
            "gmean_std":  float(np.std(gmeans)),
            "repeats":    all_results,
        }
        fname = os.path.join(
            args.save_dir,
            f"cross_{''.join(args.train_sets)}_{args.test_set}.json"
        )
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {fname}")


# ---------------------------------------------------------------------------
# K-fold cross-validation runner
# ---------------------------------------------------------------------------

def run_kfold(args) -> None:
    """
    Runs stratified k-fold cross-validation on the mixed dataset.
    Reproduces paper Table V (5-fold and 10-fold).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\n  Device  : {device}")
    print(f"  K-folds : {args.k_folds}")
    print(f"  Datasets: Ga + Ju + Si (mixed)")

    full_ds = build_kfold_datasets(
        data_root     = args.data_root,
        dataset_names = ["Ga", "Ju", "Si"],
        T             = args.T,
        aug_factor    = args.aug_factor,
        seed          = args.seed,
    )

    # Extract labels for stratified splitting
    all_labels = [full_ds[i]["label"].item() for i in range(len(full_ds))]

    skf = StratifiedKFold(
        n_splits=args.k_folds, shuffle=True, random_state=args.seed
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(range(len(full_ds)), all_labels)
    ):
        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/{args.k_folds}")
        print(f"{'='*60}")

        # Val split from train (80/20)
        n_val   = int(len(train_idx) * 0.2)
        rng     = np.random.default_rng(args.seed + fold)
        rng.shuffle(train_idx)
        val_idx   = train_idx[:n_val]
        train_idx = train_idx[n_val:]

        def make_loader(indices, shuffle):
            return DataLoader(
                Subset(full_ds, indices),
                batch_size  = args.batch_size,
                shuffle     = shuffle,
                num_workers = 0,
                pin_memory  = torch.cuda.is_available(),
            )

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader   = make_loader(val_idx,   shuffle=False)
        test_loader  = make_loader(test_idx,  shuffle=False)

        model    = GLD2GNN(T=args.T, device=device).to(device)
        run_name = f"kfold{args.k_folds}_fold{fold+1}"

        result = train(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
            epochs       = args.epochs,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            patience     = args.patience,
            use_amp      = args.amp,
            save_dir     = args.save_dir,
            run_name     = run_name,
            verbose      = True,
        )
        fold_results.append(result)

    # Aggregate
    accs   = [r["test_acc"]   for r in fold_results]
    f1s    = [r["test_f1"]    for r in fold_results]
    gmeans = [r["test_gmean"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  {args.k_folds}-Fold Cross-Validation Results")
    print(f"{'='*60}")
    print(f"  Acc  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
    print(f"  F1   : {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")
    print(f"  Gmean: {np.mean(gmeans)*100:.2f} ± {np.std(gmeans)*100:.2f}")

    if args.save_dir:
        out = {
            "mode":       f"{args.k_folds}fold_cv",
            "acc_mean":   float(np.mean(accs)),
            "acc_std":    float(np.std(accs)),
            "f1_mean":    float(np.mean(f1s)),
            "f1_std":     float(np.std(f1s)),
            "gmean_mean": float(np.mean(gmeans)),
            "gmean_std":  float(np.std(gmeans)),
            "folds":      fold_results,
        }
        fname = os.path.join(
            args.save_dir, f"kfold{args.k_folds}_results.json"
        )
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train GLD²-GNN for Parkinson's Disease detection"
    )

    # Mode
    p.add_argument("--mode", choices=["cross", "kfold"], default="cross",
                   help="cross=cross-dataset validation, kfold=k-fold CV")

    # Data
    p.add_argument("--data_root",   type=str, default="./data")
    p.add_argument("--train_sets",  nargs="+", default=["Ga", "Ju"],
                   help="Datasets for training (cross mode only)")
    p.add_argument("--test_set",    type=str,  default="Si",
                   help="Dataset for testing (cross mode only)")
    p.add_argument("--k_folds",     type=int,  default=5,
                   help="Number of folds (kfold mode only)")
    p.add_argument("--T",           type=int,  default=128,
                   help="Gait-cycle length (must be divisible by 8)")
    p.add_argument("--aug_factor",  type=int,  default=10,
                   help="Augmentation multiplier (applied to training only)")

    # Training
    p.add_argument("--epochs",       type=int,   default=120)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience",     type=int,   default=30)
    p.add_argument("--repeats",      type=int,   default=4,
                   help="Repetitions for cross-dataset (paper repeats 4×)")
    p.add_argument("--seed",         type=int,   default=42)

    # Hardware
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--amp",     action="store_true",
                   help="Mixed-precision training (RTX GPU recommended)")

    # Output
    p.add_argument("--save_dir", type=str, default="./checkpoints",
                   help="Directory for checkpoints and result JSON files")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("  GLD²-GNN  —  Parkinson's Disease Detection")
    print("=" * 60)
    print(f"  Mode        : {args.mode}")
    print(f"  T           : {args.T}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Patience    : {args.patience}")
    print(f"  AMP         : {args.amp}")
    print(f"  Save dir    : {args.save_dir}")

    if args.mode == "cross":
        run_cross_dataset(args)
    else:
        run_kfold(args)
