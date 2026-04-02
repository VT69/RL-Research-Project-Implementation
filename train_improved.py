"""
train_improved.py
=================
Training loop for GLD²-GNN+ — the improved model.

Differences from train.py
--------------------------
1. Loss = BCE + λ * sparsity_loss  (graph regularisation)
2. Uncertainty metrics logged per epoch (mean std across val set)
3. Saves side-by-side comparison JSON vs. baseline for dashboard
4. Logs per-sample alpha values to show adaptive fusion is working
5. Calibration curve data saved for dashboard uncertainty page

Usage
-----
  # Cross-dataset with improved model:
  python train_improved.py \
      --data_root ./data \
      --mode      cross \
      --train_sets Ga Ju \
      --test_set   Si \
      --epochs     120 \
      --batch_size 64

  # K-fold:
  python train_improved.py \
      --data_root ./data \
      --mode   kfold \
      --k_folds 5
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
from sklearn.model_selection import StratifiedKFold

from data_loader     import build_dataloaders, build_kfold_datasets
from model_improved  import GLD2GNNPlus
from train           import compute_metrics, vote_predict   # reuse metric helpers


# ---------------------------------------------------------------------------
# Training epoch (improved — adds sparsity loss)
# ---------------------------------------------------------------------------

def train_epoch_improved(
    model:        GLD2GNNPlus,
    loader:       DataLoader,
    optimizer:    optim.Optimizer,
    device:       torch.device,
    scaler:       Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    One training epoch for GLD²-GNN+.
    Returns dict with 'total_loss', 'bce_loss', 'sparsity_loss'.
    """
    model.train()
    totals = {"total": 0.0, "bce": 0.0, "sparsity": 0.0}
    n = 0

    for batch in loader:
        ts_nodes = batch["ts_nodes"].to(device)
        ts_edges = batch["ts_edges"].to(device)
        ms_nodes = batch["ms_nodes"].to(device)
        ms_edges = batch["ms_edges"].to(device)
        labels   = batch["label"].float().to(device).unsqueeze(1)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out  = model(ts_nodes, ts_edges, ms_nodes, ms_edges, labels)
                loss = out["bce_loss"] + out["sparsity_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(ts_nodes, ts_edges, ms_nodes, ms_edges, labels)
            loss = out["bce_loss"] + out["sparsity_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = labels.size(0)
        totals["total"]    += loss.item()           * bs
        totals["bce"]      += out["bce_loss"].item() * bs
        totals["sparsity"] += out["sparsity_loss"].item() * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Uncertainty-aware evaluation
# ---------------------------------------------------------------------------

def evaluate_with_uncertainty(
    model:     GLD2GNNPlus,
    loader:    DataLoader,
    device:    torch.device,
    n_passes:  int = 30,
    threshold: float = 0.5,
) -> Dict:
    """
    Runs MC Dropout inference on the entire loader.

    Returns:
        metrics     : acc, f1, gmean
        mean_std    : average uncertainty across samples
        alpha_vals  : list of per-sample alpha values
        all_probs   : raw mean probabilities
        all_labels  : ground truth
        calibration : dict for reliability diagram
    """
    model.eval()

    all_means, all_stds, all_true = [], [], []
    all_alphas = []

    with torch.no_grad():
        for batch in loader:
            ts_nodes = batch["ts_nodes"].to(device)
            ts_edges = batch["ts_edges"].to(device)
            ms_nodes = batch["ms_nodes"].to(device)
            ms_edges = batch["ms_edges"].to(device)
            labels   = batch["label"]

            result = model.predict_with_uncertainty(
                ts_nodes, ts_edges, ms_nodes, ms_edges,
                n_passes=n_passes, threshold=threshold,
            )

            all_means.extend(result["mean"].cpu().tolist())
            all_stds.extend(result["std"].cpu().tolist())
            all_true.extend(labels.tolist())
            all_alphas.extend(result["alpha"].cpu().tolist())

    y_true = np.array(all_true)
    y_pred = (np.array(all_means) >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred)

    # Calibration data for reliability diagram (10 bins)
    bins       = np.linspace(0, 1, 11)
    bin_accs   = []
    bin_confs  = []
    bin_counts = []
    means_arr  = np.array(all_means)
    for i in range(10):
        mask = (means_arr >= bins[i]) & (means_arr < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(float(y_true[mask].mean()))
            bin_confs.append(float(means_arr[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bins[i] + bins[i+1]) / 2))
            bin_counts.append(0)

    return {
        "metrics":     metrics,
        "mean_std":    float(np.mean(all_stds)),
        "alpha_vals":  all_alphas,
        "all_probs":   all_means,
        "all_stds":    all_stds,
        "all_labels":  all_true,
        "calibration": {
            "bin_accs":   bin_accs,
            "bin_confs":  bin_confs,
            "bin_counts": bin_counts,
        },
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_improved(
    model:        GLD2GNNPlus,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 120,
    lr:           float = 5e-4,
    weight_decay: float = 5e-4,
    patience:     int   = 30,
    n_mc_passes:  int   = 30,
    use_amp:      bool  = False,
    save_dir:     Optional[str] = None,
    run_name:     str   = "improved_run",
    verbose:      bool  = True,
) -> Dict:
    """
    Full training loop for GLD²-GNN+ with early stopping and cosine LR decay.

    Key differences from train.py::train()
    ----------------------------------------
    * Loss = BCE + λ * sparsity_loss  (graph L1 regularisation, GAP 3)
    * Val metrics use MC Dropout (n_passes=10 during training for speed)
    * Logs per-epoch uncertainty mean and per-sample fusion weights
    * Checkpoint saves model.state_dict() at best val_loss

    Args:
        model        : GLD2GNNPlus instance (moved to device internally)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        test_loader  : test DataLoader (evaluated once with n_mc_passes)
        device       : torch device
        epochs       : maximum training epochs (paper: 120)
        lr           : initial learning rate (paper: 5e-4)
        weight_decay : L2 regularisation coefficient (paper: 5e-4)
        patience     : early-stopping patience in epochs (paper: 30)
        n_mc_passes  : MC Dropout passes for final test evaluation
        use_amp      : enable mixed-precision (recommended on RTX GPUs)
        save_dir     : directory for checkpoints (None = no saving)
        run_name     : filename prefix for saved checkpoint
        verbose      : print per-epoch metrics table

    Returns:
        dict with keys: best_epoch, test_acc, test_f1, test_gmean,
                        mean_uncertainty, alpha_vals, calibration,
                        all_probs, all_stds, all_labels, history
    """
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        betas=(0.9, 0.995), weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=14, eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_path = os.path.join(save_dir, f"{run_name}_best.pt")
    else:
        best_path = None

    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = 0

    # NOTE: val_loss is read directly from out["bce_loss"] + out["sparsity_loss"]
    # returned by the model forward pass — no separate criterion object needed here.

    history = {
        "train_total_loss": [], "train_bce_loss": [], "train_sparsity_loss": [],
        "val_acc": [], "val_f1": [], "val_gmean": [],
        "val_mean_uncertainty": [],
    }

    if verbose:
        print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'BCE':>7}  {'Sparse':>8}  "
              f"{'VaAcc':>7}  {'VaF1':>7}  {'VaGm':>7}  {'Uncert':>7}")
        print("-" * 72)

    for epoch in range(1, epochs + 1):

        # Train
        losses = train_epoch_improved(
            model, train_loader, optimizer, device, scaler
        )

        # Validate with MC Dropout
        val_result = evaluate_with_uncertainty(
            model, val_loader, device, n_passes=10  # fewer passes during training
        )
        val_m = val_result["metrics"]

        # Val loss for early stopping — normalise by number of samples,
        # consistent with train_epoch_improved which also aggregates per sample.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    batch["ts_nodes"].to(device),
                    batch["ts_edges"].to(device),
                    batch["ms_nodes"].to(device),
                    batch["ms_edges"].to(device),
                    batch["label"].float().to(device).unsqueeze(1),
                )
                bs       = batch["label"].size(0)
                val_loss += (out["bce_loss"] + out["sparsity_loss"]).item() * bs
        val_loss /= len(val_loader.dataset)   # per-sample average

        scheduler.step()
        model.step()

        # History
        history["train_total_loss"].append(losses["total"])
        history["train_bce_loss"].append(losses["bce"])
        history["train_sparsity_loss"].append(losses["sparsity"])
        history["val_acc"].append(val_m["acc"])
        history["val_f1"].append(val_m["f1"])
        history["val_gmean"].append(val_m["gmean"])
        history["val_mean_uncertainty"].append(val_result["mean_std"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_epoch = epoch
            if best_path:
                torch.save({
                    "epoch": epoch, "state_dict": model.state_dict(),
                    "val_acc": val_m["acc"], "optimizer": optimizer.state_dict(),
                }, best_path)
        else:
            patience_count += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"{epoch:>4}  {losses['total']:>8.4f}  {losses['bce']:>7.4f}  "
                f"{losses['sparsity']:>8.6f}  {val_m['acc']:>7.4f}  "
                f"{val_m['f1']:>7.4f}  {val_m['gmean']:>7.4f}  "
                f"{val_result['mean_std']:>7.4f}  "
                f"{'*' if patience_count == 0 else ''}"
            )

        if patience_count >= patience:
            if verbose:
                print(f"\n  Early stop at epoch {epoch} (best={best_epoch})")
            break

    # Load best and do full MC evaluation on test set
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    test_result = evaluate_with_uncertainty(
        model, test_loader, device, n_passes=n_mc_passes
    )
    test_m = test_result["metrics"]

    if verbose:
        print(f"\n  Test results (MC Dropout, {n_mc_passes} passes):")
        print(f"    Acc       = {test_m['acc']:.4f}")
        print(f"    F1        = {test_m['f1']:.4f}")
        print(f"    Gmean     = {test_m['gmean']:.4f}")
        print(f"    Mean uncertainty = {test_result['mean_std']:.4f}")
        print(f"    Mean alpha (fusion weight) = {np.mean(test_result['alpha_vals']):.4f}")

    return {
        "best_epoch":       best_epoch,
        "test_acc":         test_m["acc"],
        "test_f1":          test_m["f1"],
        "test_gmean":       test_m["gmean"],
        "mean_uncertainty": test_result["mean_std"],
        "alpha_vals":       test_result["alpha_vals"],
        "calibration":      test_result["calibration"],
        "all_probs":        test_result["all_probs"],
        "all_stds":         test_result["all_stds"],
        "all_labels":       test_result["all_labels"],
        "history":          history,
    }


# ---------------------------------------------------------------------------
# Comparison runner — runs both models and saves side-by-side JSON
# ---------------------------------------------------------------------------

def run_comparison(args) -> None:
    """
    Trains both GLD2GNN (baseline) and GLD2GNNPlus (improved) under
    identical conditions and saves a comparison JSON for the dashboard.
    """
    from model import GLD2GNN
    from train import train as train_baseline

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\n  Device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    all_baseline, all_improved = [], []

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        print(f"\n{'='*65}")
        print(f"  Repeat {repeat+1}/{args.repeats}  seed={seed}")
        print(f"{'='*65}")

        train_loader, val_loader, test_loader = build_dataloaders(
            data_root=args.data_root, train_sets=args.train_sets,
            test_set=args.test_set, T=args.T,
            aug_factor=args.aug_factor, batch_size=args.batch_size,
            val_split=0.2, seed=seed,
        )

        # --- Baseline ---
        print(f"\n  [Baseline GLD²-GNN]")
        baseline = GLD2GNN(T=args.T, device=device).to(device)
        b_result = train_baseline(
            model=baseline, train_loader=train_loader,
            val_loader=val_loader, test_loader=test_loader,
            device=device, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            use_amp=args.amp,
            save_dir=args.save_dir, run_name=f"baseline_r{repeat}",
            verbose=True,
        )
        all_baseline.append(b_result)

        # --- Improved ---
        print(f"\n  [Improved GLD²-GNN+]")
        improved = GLD2GNNPlus(T=args.T, device=device).to(device)
        i_result = train_improved(
            model=improved, train_loader=train_loader,
            val_loader=val_loader, test_loader=test_loader,
            device=device, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            use_amp=args.amp,
            save_dir=args.save_dir, run_name=f"improved_r{repeat}",
            verbose=True,
        )
        all_improved.append(i_result)

    def agg(results, key):
        vals = [r[key] for r in results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    comparison = {
        "mode": "cross_dataset",
        "train_sets": args.train_sets,
        "test_set": args.test_set,
        "baseline": {
            "acc":   agg(all_baseline, "test_acc"),
            "f1":    agg(all_baseline, "test_f1"),
            "gmean": agg(all_baseline, "test_gmean"),
        },
        "improved": {
            "acc":   agg(all_improved, "test_acc"),
            "f1":    agg(all_improved, "test_f1"),
            "gmean": agg(all_improved, "test_gmean"),
            "mean_uncertainty": agg(all_improved, "mean_uncertainty"),
        },
        "raw_baseline": all_baseline,
        "raw_improved": all_improved,
    }

    fname = os.path.join(args.save_dir, "comparison_results.json")
    with open(fname, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*65}")
    for model_name, key in [("Baseline", "baseline"), ("Improved", "improved")]:
        d = comparison[key]
        print(f"\n  {model_name}:")
        print(f"    Acc  : {d['acc']['mean']*100:.2f} ± {d['acc']['std']*100:.2f}")
        print(f"    F1   : {d['f1']['mean']*100:.2f} ± {d['f1']['std']*100:.2f}")
        print(f"    Gmean: {d['gmean']['mean']*100:.2f} ± {d['gmean']['std']*100:.2f}")
    print(f"\n  Saved → {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train GLD²-GNN+ (improved model)")
    p.add_argument("--mode",         choices=["cross", "kfold", "compare"], default="compare")
    p.add_argument("--data_root",    type=str, default="./data")
    p.add_argument("--train_sets",   nargs="+", default=["Ga", "Ju"])
    p.add_argument("--test_set",     type=str,  default="Si")
    p.add_argument("--k_folds",      type=int,  default=5)
    p.add_argument("--T",            type=int,  default=128)
    p.add_argument("--aug_factor",   type=int,  default=10)
    p.add_argument("--epochs",       type=int,  default=120)
    p.add_argument("--batch_size",   type=int,  default=64)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience",     type=int,  default=30)
    p.add_argument("--repeats",      type=int,  default=4)
    p.add_argument("--n_mc_passes",  type=int,  default=30)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--no_cuda",      action="store_true")
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--save_dir",     type=str,  default="./checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 65)
    print("  GLD²-GNN+  —  Improved Parkinson's Disease Detection")
    print("=" * 65)
    print(f"  Mode        : {args.mode}")
    print(f"  T           : {args.T}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Patience    : {args.patience}")
    print(f"  AMP         : {args.amp}")
    print(f"  Save dir    : {args.save_dir}")

    if args.mode == "compare":
        # Train baseline + improved side by side and save comparison JSON
        run_comparison(args)

    elif args.mode == "cross":
        # Cross-dataset validation with the improved model only
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        print(f"\n  Device : {device}")
        print(f"  Train  : {args.train_sets}  →  Test: {args.test_set}")
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        all_results = []
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            print(f"\n{'='*65}\n  Repeat {repeat+1}/{args.repeats}  (seed={seed})\n{'='*65}")
            train_loader, val_loader, test_loader = build_dataloaders(
                data_root=args.data_root, train_sets=args.train_sets,
                test_set=args.test_set, T=args.T,
                aug_factor=args.aug_factor, batch_size=args.batch_size,
                val_split=0.2, seed=seed,
            )
            model = GLD2GNNPlus(T=args.T, device=device).to(device)
            result = train_improved(
                model=model, train_loader=train_loader,
                val_loader=val_loader, test_loader=test_loader,
                device=device, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience,
                n_mc_passes=args.n_mc_passes, use_amp=args.amp,
                save_dir=args.save_dir,
                run_name=f"improved_r{repeat}", verbose=True,
            )
            all_results.append(result)

        accs   = [r["test_acc"]   for r in all_results]
        f1s    = [r["test_f1"]    for r in all_results]
        gmeans = [r["test_gmean"] for r in all_results]
        import numpy as np
        print(f"\n{'='*65}\n  GLD²-GNN+ Cross-Dataset Results\n{'='*65}")
        print(f"  Acc  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
        print(f"  F1   : {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")
        print(f"  Gmean: {np.mean(gmeans)*100:.2f} ± {np.std(gmeans)*100:.2f}")

    elif args.mode == "kfold":
        # K-fold cross-validation with the improved model only
        from torch.utils.data import Subset
        from sklearn.model_selection import StratifiedKFold
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        print(f"\n  Device  : {device}\n  K-folds : {args.k_folds}")
        full_ds = build_kfold_datasets(
            data_root=args.data_root, dataset_names=["Ga", "Ju", "Si"],
            T=args.T, aug_factor=args.aug_factor, seed=args.seed,
        )
        all_labels = [full_ds[i]["label"].item() for i in range(len(full_ds))]
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True,
                              random_state=args.seed)
        import numpy as np
        all_results = []
        for fold, (train_idx, test_idx) in enumerate(
            skf.split(range(len(full_ds)), all_labels)
        ):
            print(f"\n{'='*65}\n  Fold {fold+1}/{args.k_folds}\n{'='*65}")
            n_val     = int(len(train_idx) * 0.2)
            rng       = np.random.default_rng(args.seed + fold)
            rng.shuffle(train_idx)
            val_idx   = train_idx[:n_val]
            train_idx = train_idx[n_val:]

            def _make_loader(idx, shuffle):
                return DataLoader(
                    Subset(full_ds, idx), batch_size=args.batch_size,
                    shuffle=shuffle, num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )

            model = GLD2GNNPlus(T=args.T, device=device).to(device)
            result = train_improved(
                model=model,
                train_loader=_make_loader(train_idx, True),
                val_loader=_make_loader(val_idx, False),
                test_loader=_make_loader(test_idx, False),
                device=device, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience,
                n_mc_passes=args.n_mc_passes, use_amp=args.amp,
                save_dir=args.save_dir,
                run_name=f"improved_kfold{args.k_folds}_fold{fold+1}",
                verbose=True,
            )
            all_results.append(result)

        accs   = [r["test_acc"]   for r in all_results]
        f1s    = [r["test_f1"]    for r in all_results]
        gmeans = [r["test_gmean"] for r in all_results]
        print(f"\n{'='*65}\n  GLD²-GNN+  {args.k_folds}-Fold CV Results\n{'='*65}")
        print(f"  Acc  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
        print(f"  F1   : {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")
        print(f"  Gmean: {np.mean(gmeans)*100:.2f} ± {np.std(gmeans)*100:.2f}")
