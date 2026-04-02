"""
data_loader.py
==============
Handles everything between raw PhysioNet files and model-ready tensors.

Pipeline
--------
1. Load raw .txt VGRF files from PhysioNet (Ga / Ju / Si datasets).
2. Segment each recording into individual gait-cycle samples.
3. Augment via window-slicing + permutation (paper Section IV-A-1).
4. Build the two-stream sequences:
      time stream   L       = raw node values
      motion stream dif(L)  = frame-to-frame differences
5. Compute directed edge values from node features (eq. 4).
6. Return PyTorch Dataset / DataLoader objects ready for training.

PhysioNet folder structure expected
------------------------------------
  data_root/
    Ga/
      GaXXXXX_<condition>.txt    (condition: "pd" or "co")
    Ju/
      JuXXXXX_<condition>.txt
    Si/
      SiXXXXX_<condition>.txt

Each .txt file: whitespace-separated, columns =
  [time, L1, L2, ..., L8, R1, R2, ..., R8, total_L, total_R]
  (19 columns total; we use columns 1-16 = the 16 individual sensors)

Usage
-----
  from data_loader import build_dataloaders

  train_loader, val_loader, test_loader = build_dataloaders(
      data_root   = "path/to/physionet",
      train_sets  = ["Ga", "Ju"],
      test_set    = "Si",
      T           = 128,       # target gait-cycle length after interpolation
      s_first     = 4,         # time slices in first DyDGNN block
      aug_factor  = 10,        # augmentation multiplications
      batch_size  = 64,
      val_split   = 0.2,
      seed        = 42,
  )
"""

import os
import re
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from graph_construction import compute_edge_values, NV


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NE       = 26    # number of directed edges (matches graph_construction.py)
DATASETS = ["Ga", "Ju", "Si"]


# ---------------------------------------------------------------------------
# 1.  Raw file loading
# ---------------------------------------------------------------------------

def _label_from_filename(fname: str) -> int:
    """
    Infers class label from a PhysioNet gaitpdb filename.

    PhysioNet naming convention (case-insensitive):
        <Dataset>Pt<ID>_<trial>.txt  →  PD patient   → label 1
        <Dataset>Co<ID>_<trial>.txt  →  Healthy ctrl  → label 0

    Examples::
        GaPt03_01  → 1  (Ga dataset, PD patient 03, trial 01)
        GaCo01_01  → 0  (Ga dataset, healthy control 01, trial 01)
        JuPt12_02  → 1
        SiCo05_01  → 0

    Args:
        fname: filename stem (no extension)

    Returns:
        0 for healthy control, 1 for PD patient

    Raises:
        ValueError if the filename does not match the expected pattern.
    """
    # Use regex anchored to the dataset prefix to avoid false positives.
    # Pattern: optional dataset prefix (Ga/Ju/Si), then Pt or Co, then digits.
    if re.search(r"(?:ga|ju|si)?pt\d", fname, re.IGNORECASE):
        return 1
    elif re.search(r"(?:ga|ju|si)?co\d", fname, re.IGNORECASE):
        return 0
    else:
        raise ValueError(
            f"Cannot infer label from filename '{fname}'. "
            "Expected PhysioNet gaitpdb pattern: <Dataset>Pt<ID>_<trial> "
            "or <Dataset>Co<ID>_<trial>  (e.g. GaPt03_01 or SiCo05_01)."
        )


def load_raw_records(data_root: str, dataset_name: str) -> List[Dict]:
    """
    Loads all .txt records for one dataset (Ga, Ju, or Si).

    Returns
    -------
    List of dicts, each with keys:
        'signal' : np.ndarray  [T_raw, 16]  float32  (16 sensor columns)
        'label'  : int   0=CO, 1=PD
        'subject': str   filename stem
        'dataset': str   dataset name
    """
    folder = Path(data_root) / dataset_name
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Download from https://physionet.org/files/gaitpdb/1.0.0/"
        )

    records = []
    for fpath in sorted(folder.glob("*.txt")):
        try:
            label = _label_from_filename(fpath.stem)
        except ValueError:
            continue  # skip files with ambiguous names

        raw = np.loadtxt(fpath, dtype=np.float32)   # [T_raw, 18+]

        # Columns 1-16 (0-indexed) are the 16 individual sensors
        # Column 0 is the timestamp; columns 17-19 are totals — skip them
        if raw.ndim == 1:
            continue  # malformed / empty file
        if raw.shape[1] < 19:  # format.txt: 19 cols (time + 8L + 8R + totalL + totalR)
            continue

        signal = raw[:, 1:17]   # [T_raw, 16]

        records.append({
            "signal":  signal,
            "label":   label,
            "subject": fpath.stem,
            "dataset": dataset_name,
        })

    if not records:
        raise RuntimeError(f"No valid records found in {folder}")

    return records


# ---------------------------------------------------------------------------
# 2.  Gait-cycle segmentation
# ---------------------------------------------------------------------------

def _find_segmentation_points(signal: np.ndarray) -> List[int]:
    """
    Finds valid segmentation points for gait cycles.

    Per the paper: a segmentation point is selected from a frame that is
    smaller than the first 40 points AND not within the last 40 points.
    We use the total force (sum across sensors) local minima as candidates,
    consistent with standard gait analysis practice.

    Returns a list of frame indices where gait cycles begin.
    """
    total_force = signal.sum(axis=1)   # [T_raw]
    T = len(total_force)

    candidates = []
    for i in range(40, T - 40):
        # Local minimum: smaller than 40 neighbours on each side
        window = total_force[max(0, i-5): i+6]
        if total_force[i] == window.min():
            candidates.append(i)

    # Deduplicate: keep only those separated by at least 40 frames
    filtered = []
    last = -100
    for c in candidates:
        if c - last >= 40:
            filtered.append(c)
            last = c

    return filtered


def segment_gait_cycles(signal: np.ndarray) -> List[np.ndarray]:
    """
    Splits a raw VGRF recording into individual gait-cycle segments.

    Args:
        signal: [T_raw, 16]

    Returns:
        List of variable-length segments, each of shape [T_cycle, 16].
    """
    points = _find_segmentation_points(signal)

    if len(points) < 2:
        return []  # too short to segment

    cycles = []
    for i in range(len(points) - 1):
        seg = signal[points[i]: points[i+1]]
        if len(seg) >= 20:   # discard very short artefact cycles
            cycles.append(seg)

    return cycles


# ---------------------------------------------------------------------------
# 3.  Resize to fixed length T via cubic spline interpolation
# ---------------------------------------------------------------------------

def resize_cycle(cycle: np.ndarray, T: int) -> np.ndarray:
    """
    Resamples a variable-length gait cycle to exactly T frames using
    cubic spline interpolation (paper Section IV-A-1).

    Args:
        cycle: [T_cycle, 16]
        T    : target length

    Returns:
        [T, 16] float32 array
    """
    T_cycle = len(cycle)
    if T_cycle == T:
        return cycle.astype(np.float32)

    x_old = np.linspace(0, 1, T_cycle)
    x_new = np.linspace(0, 1, T)

    cs = CubicSpline(x_old, cycle, axis=0)
    return cs(x_new).astype(np.float32)


# ---------------------------------------------------------------------------
# 4.  Data augmentation
# ---------------------------------------------------------------------------

def window_slice(cycle: np.ndarray, T: int, w_ratio: float = 0.9) -> np.ndarray:
    """
    Window-slicing augmentation (paper Section IV-A-1).
    Crops a random contiguous sub-window of length w = w_ratio * T,
    then resamples back to T via cubic spline — analogous to image cropping.

    Args:
        cycle  : [T, 16]
        T      : target output length
        w_ratio: fraction of T to crop (default 0.9)

    Returns:
        [T, 16] augmented sample
    """
    w = max(int(T * w_ratio), 10)
    start = random.randint(0, T - w)
    cropped = cycle[start: start + w]
    return resize_cycle(cropped, T)


def permutation_augment(cycle: np.ndarray) -> np.ndarray:
    """
    Permutation augmentation (paper Section IV-A-1).
    Randomly selects a split axis along the gait cycle and swaps
    the left/right halves WITHOUT changing the plantar data values.

    Args:
        cycle: [T, 16]

    Returns:
        [T, 16] permuted sample
    """
    T = len(cycle)
    split = random.randint(1, T - 1)
    return np.concatenate([cycle[split:], cycle[:split]], axis=0)


class LazyAugmentedSample:
    """
    Lightweight reference to a base cycle + augmentation mode.
    Prevents 10x memory explosion by delaying augmentation to exactly
    when the DataLoader requests the item in __getitem__.
    """
    def __init__(self, base_cycle: np.ndarray, mode: int):
        self.base_cycle = base_cycle
        self.mode = mode

    def get(self, T: int) -> np.ndarray:
        """
        Applies the stored augmentation mode to base_cycle.

        Modes
        -----
        -1 : no augmentation — return original cycle as-is
         0 : window-slice only
         1 : permutation only
         2 : permutation + window-slice (combined)
        """
        if self.mode == -1:
            # Original, unaugmented sample
            return self.base_cycle
        elif self.mode == 0:
            return window_slice(self.base_cycle, T)
        elif self.mode == 1:
            return permutation_augment(self.base_cycle)
        elif self.mode == 2:
            return permutation_augment(window_slice(self.base_cycle, T))
        else:
            raise ValueError(
                f"LazyAugmentedSample: unknown augmentation mode {self.mode!r}. "
                "Expected -1 (none), 0 (window-slice), 1 (permutation), or 2 (both)."
            )


def augment_samples(
    cycles: List[np.ndarray],
    T: int,
    aug_factor: int = 10,
) -> List[LazyAugmentedSample]:
    """
    Returns lightweight references instead of computing augmentations immediately.
    """
    augmented = []

    for cycle in cycles:
        augmented.append(LazyAugmentedSample(cycle, -1))  # -1 means Original
        for _ in range(aug_factor):
            mode = random.randint(0, 2)
            augmented.append(LazyAugmentedSample(cycle, mode))

    return augmented


# ---------------------------------------------------------------------------
# 5.  Two-stream construction
# ---------------------------------------------------------------------------

def build_two_streams(
    node_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the time stream and motion stream from a node sequence.

    Time stream   L       = node_seq  (raw values)
    Motion stream dif(L)  = frame-to-frame differences (equation 5)

    The motion stream has the same shape as the time stream.
    The first frame difference is set to zero (no prior frame).

    Args:
        node_seq: [T, NV]  float32

    Returns:
        time_stream  : [T, NV]
        motion_stream: [T, NV]
    """
    time_stream   = node_seq.copy()
    motion_stream = np.zeros_like(node_seq)
    motion_stream[1:] = node_seq[1:] - node_seq[:-1]

    return time_stream, motion_stream


# Cache edge indices to avoid 1M+ import/loop operations
_EDGES = None
_SRC_IDX = None
_DST_IDX = None

def build_edge_sequence(node_seq: np.ndarray) -> np.ndarray:
    """
    Computes edge features from node sequence using equation (4).
    """
    global _EDGES, _SRC_IDX, _DST_IDX
    if _EDGES is None:
        from graph_construction import _build_edge_list
        _EDGES   = _build_edge_list()
        _SRC_IDX = [e[0] for e in _EDGES]
        _DST_IDX = [e[1] for e in _EDGES]

    v_src = node_seq[:, _SRC_IDX]   # [T, NE]
    v_dst = node_seq[:, _DST_IDX]   # [T, NE]
    return (v_dst - v_src).astype(np.float32)


# ---------------------------------------------------------------------------
# 6.  PyTorch Dataset
# ---------------------------------------------------------------------------

class VGRFDataset(Dataset):
    """
    PyTorch Dataset for GLD2-GNN.

    Each item is a dict with:
        'ts_nodes' : [1, T, NV]  time-stream  node features
        'ts_edges' : [1, T, NE]  time-stream  edge features
        'ms_nodes' : [1, T, NV]  motion-stream node features
        'ms_edges' : [1, T, NE]  motion-stream edge features
        'label'    : scalar int  (0=CO, 1=PD)

    The channel dimension (1) is the initial C_in for the first DyDGNN block.
    """

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],
        T: int,
    ):
        """
        Args:
            samples: list of (node_seq [T, NV], label) tuples
            T      : fixed gait-cycle length
        """
        self.T       = T
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]

        # Evaluate lazy augmentation if needed
        if isinstance(item, LazyAugmentedSample):
            node_seq = item.get(self.T)
        else:
            node_seq = item

        ts_nodes, ms_nodes = build_two_streams(node_seq)
        ts_edges = build_edge_sequence(ts_nodes)
        ms_edges = build_edge_sequence(ms_nodes)

        # Add channel dim: [T, NV] → [1, T, NV]
        return {
            "ts_nodes": torch.from_numpy(ts_nodes).unsqueeze(0),  # [1,T,NV]
            "ts_edges": torch.from_numpy(ts_edges).unsqueeze(0),  # [1,T,NE]
            "ms_nodes": torch.from_numpy(ms_nodes).unsqueeze(0),  # [1,T,NV]
            "ms_edges": torch.from_numpy(ms_edges).unsqueeze(0),  # [1,T,NE]
            "label":    torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 7.  High-level builder
# ---------------------------------------------------------------------------

def _load_and_prepare(
    data_root: str,
    dataset_names: List[str],
    T: int,
    aug_factor: int,
    augment: bool = True,
    seed: int = 42,
) -> List[Tuple[np.ndarray, int]]:
    """
    Loads, segments, resizes, and (optionally) augments records from
    one or more named datasets.

    Returns a flat list of (node_seq [T, NV], label) tuples.
    """
    random.seed(seed)
    np.random.seed(seed)

    all_samples = []

    for ds_name in dataset_names:
        records = load_raw_records(data_root, ds_name)

        for rec in records:
            cycles = segment_gait_cycles(rec["signal"])
            if not cycles:
                continue

            resized = [resize_cycle(c, T) for c in cycles]

            if augment:
                resized_items = augment_samples(resized, T, aug_factor=aug_factor)
            else:
                resized_items = resized

            for item in resized_items:
                all_samples.append((item, rec["label"]))

    return all_samples


def build_dataloaders(
    data_root:   str,
    train_sets:  List[str],
    test_set:    str,
    T:           int   = 128,
    aug_factor:  int   = 10,
    batch_size:  int   = 64,
    val_split:   float = 0.2,
    seed:        int   = 42,
    num_workers: int   = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Main entry point.  Builds train / val / test DataLoaders.

    For cross-dataset validation (paper Section IV-A-2):
        train_sets = ["Ga", "Ju"],  test_set = "Si"  (or any combination)

    For mixed-data k-fold, use build_kfold_datasets() below instead.

    Args:
        data_root  : path to PhysioNet data root
        train_sets : list of dataset names for training  (e.g. ["Ga","Ju"])
        test_set   : dataset name for testing            (e.g. "Si")
        T          : gait-cycle length after interpolation
        aug_factor : augmentation multiplier (applied to train only)
        batch_size : mini-batch size
        val_split  : fraction of training data for validation
        seed       : random seed for reproducibility
        num_workers: DataLoader worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    torch.manual_seed(seed)

    # Training data (augmented)
    train_samples = _load_and_prepare(
        data_root, train_sets, T, aug_factor, augment=True, seed=seed
    )

    # Test data (no augmentation)
    test_samples = _load_and_prepare(
        data_root, [test_set], T, aug_factor=0, augment=False, seed=seed
    )

    # Split train → train + val  (80/20 as in paper)
    n_val   = int(len(train_samples) * val_split)
    n_train = len(train_samples) - n_val

    train_ds  = VGRFDataset(train_samples, T)
    test_ds   = VGRFDataset(test_samples,  T)

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_ds = random_split(
        train_ds, [n_train, n_val], generator=generator
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        make_loader(train_split, shuffle=True),
        make_loader(val_split_ds, shuffle=False),
        make_loader(test_ds,      shuffle=False),
    )


def build_kfold_datasets(
    data_root:   str,
    dataset_names: List[str],
    T:           int   = 128,
    aug_factor:  int   = 10,
    seed:        int   = 42,
) -> VGRFDataset:
    """
    Returns a single VGRFDataset from all named datasets combined,
    ready to be split by a k-fold cross-validator (e.g. sklearn's KFold).

    Usage:
        from sklearn.model_selection import KFold
        full_ds = build_kfold_datasets(data_root, ["Ga","Ju","Si"])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(range(len(full_ds))):
            ...
    """
    all_samples = _load_and_prepare(
        data_root, dataset_names, T, aug_factor, augment=True, seed=seed
    )
    return VGRFDataset(all_samples, T)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print("=" * 55)
    print("data_loader.py sanity check")
    print("=" * 55)

    # --- Mock a tiny dataset if real data not available ---
    try:
        records = load_raw_records(data_root, "Ga")
        print(f"  Loaded {len(records)} records from Ga")
        rec = records[0]
        print(f"  First record: {rec['subject']}  label={rec['label']}"
              f"  signal shape={rec['signal'].shape}")

        cycles = segment_gait_cycles(rec["signal"])
        print(f"  Gait cycles found: {len(cycles)}")

        if cycles:
            r = resize_cycle(cycles[0], T=128)
            print(f"  Resized cycle shape: {r.shape}")

            ts, ms = build_two_streams(r)
            print(f"  Time stream : {ts.shape}")
            print(f"  Motion stream: {ms.shape}")

            es = build_edge_sequence(ts)
            print(f"  Edge sequence: {es.shape}")

    except FileNotFoundError:
        print("  [INFO] Real data not found — running synthetic mock test")

        # Synthetic mock: 2 fake records
        T, NV_loc = 128, 16
        mock_signal = np.random.rand(500, NV_loc).astype(np.float32)
        mock_signal[:, :] *= 100  # simulate ground reaction forces (N)

        cycles = segment_gait_cycles(mock_signal)
        if not cycles:
            # Fallback: manually slice
            cycles = [mock_signal[i*60:(i+1)*60] for i in range(6)]

        resized = [resize_cycle(c, T) for c in cycles[:3]]
        aug     = augment_samples(resized, T, aug_factor=3)

        samples = [(s, 0) for s in aug]
        ds      = VGRFDataset(samples, T)

        item = ds[0]
        print(f"  ts_nodes : {item['ts_nodes'].shape}")
        print(f"  ts_edges : {item['ts_edges'].shape}")
        print(f"  ms_nodes : {item['ms_nodes'].shape}")
        print(f"  ms_edges : {item['ms_edges'].shape}")
        print(f"  label    : {item['label']}")

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch  = next(iter(loader))
        print(f"\n  Batch ts_nodes : {batch['ts_nodes'].shape}")
        print(f"  Batch ts_edges : {batch['ts_edges'].shape}")
        print(f"  Batch label    : {batch['label']}")

    print("\n  Sanity check complete.")
