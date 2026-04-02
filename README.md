# GLD²-GNN: Global-Local Dynamic Directed Graph Neural Network for Parkinson's Disease Detection

> **Unofficial Implementation** of the paper:
> *"A Global-Local Dynamic Directed Graph Neural Network for Parkinson's Disease Detection"*
> Xiaotian Wang, Guanhai Zhou, Zhifu Zhao, Xiaoyi Zhang, Fu Li, Fei Qi
> **IEEE Transactions on Neural Systems and Rehabilitation Engineering, Vol. 33, 2025**
> DOI: [10.1109/TNSRE.2025.3614430](https://doi.org/10.1109/TNSRE.2025.3614430)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [GLD²-GNN+ & Interactive Dashboard](#✨-gld²-gnn--interactive-dashboard)
- [Results](#results)
- [Project Structure](#project-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Dataset Setup](#dataset-setup)
- [Running Experiments](#running-experiments)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Credits & Citation](#credits--citation)
- [Disclaimer](#disclaimer)

---

## Overview

Parkinson's Disease (PD) causes measurable changes in gait patterns that can be captured through **Vertical Ground Reaction Force (VGRF)** signals — pressure data from 16 sensors embedded in shoe insoles.

This repository implements GLD²-GNN, which treats VGRF signals as **dynamic directed graphs** rather than flat grid signals. The core insight is that plantar pressure transmission paths change at every phase of the gait cycle, and static graph methods miss these critical temporal topology changes.

**Key contributions implemented:**
- **Dynamic Graph Learning (DGL) unit** — 5-branch architecture learning global and local adjacency matrices per time slot
- **Dynamic Directed Graph Network (DyDGN) unit** — aggregates node, edge, and dynamic edge features simultaneously
- **Temporal Convolutional Network (TCN) unit** — captures local gait-cycle temporal patterns with separate parameters for nodes and edges
- **Two-stream framework** — parallel time stream and motion stream with learnable α-fusion

---

## Architecture

```
VGRF Input (16 sensors)
        │
        ▼
Two-Stream Framework
   ┌────┴────┐
   │         │
Time       Motion
Stream     Stream
   │         │
   └────┬────┘
        │
  ┌─────▼──────┐  ×4 blocks
  │ DyDGNN     │
  │  ├─ DGL    │  ← learns dynamic adjacency (5 branches)
  │  ├─ DyDGN  │  ← spatial feature aggregation
  │  └─ TCN    │  ← temporal feature extraction
  └─────┬──────┘
        │
  Classification Head
  (concat node + edge features → sigmoid)
        │
   α-weighted fusion
        │
   PD / Healthy Control
```

**Channel schedule across 4 DyDGNN blocks:**

| Block | C\_in | C\_out | Time slots (s) |
|-------|-------|--------|----------------|
| 1     | 1     | 32     | 4              |
| 2     | 32    | 32     | 8              |
| 3     | 32    | 32     | 8              |
| 4     | 32    | 64     | 8              |

---

## ✨ GLD²-GNN+ & Interactive Dashboard

This repository extends the original paper with **GLD²-GNN+**, introducing targeted enhancements to address key research gaps:
1. **Adaptive Fusion**: Replaces the fixed scalar $\alpha$ fusion with an adaptive, sample-wise fusion network that dynamically weights the time and motion streams per patient and gait phase.
2. **Uncertainty Estimation**: Incorporates Monte Carlo Dropout to provide a confidence measure alongside predictions, assisting in clinical interpretation for borderline cases.

### Interactive Dashboard

An interactive Streamlit dashboard is provided to explore these improvements, visualize dynamic graphs, and perform live inference.

```bash
# Train both models for comparison
python train_improved.py --mode compare --data_root ./data --epochs 120

# Launch the Streamlit dashboard
streamlit run app.py
```

**Dashboard Features:**
- 📖 **Research Gap Analysis**: Side-by-side comparison of the paper's limitations vs. proposed fixes.
- ⚡ **Live Inference**: Upload raw VGRF `.txt` files to get predictions with uncertainty bounds.
- 🕸️ **Dynamic Graph Viewer**: Animated visualization of learned adjacency matrices across gait phases.
- 📊 **Results Comparison**: Direct metric comparisons (Accuracy, F1, G-Mean) between the baseline and GLD²-GNN+.

---

## Results

Results from the original paper (Tables IV & V) that this implementation targets:

### Cross-Dataset Validation (Table IV)

| Train → Test | Acc (%)       | F1 (%)        | G-mean (%)    |
|--------------|---------------|---------------|---------------|
| Ga+Ju → Si   | 81.25 ± 1.10  | 83.34 ± 0.82  | 83.40 ± 0.78  |
| Ga+Si → Ju   | 87.21 ± 0.87  | 92.40 ± 0.59  | 92.52 ± 0.64  |
| Si+Ju → Ga   | 81.41 ± 0.63  | 87.57 ± 0.36  | 88.14 ± 0.38  |

### Mixed-Data Cross-Validation (Table V)

| Folds   | Acc (%)       | F1 (%)        | G-mean (%)    |
|---------|---------------|---------------|---------------|
| 5-fold  | 98.48 ± 0.39  | 98.85 ± 0.29  | 98.85 ± 0.29  |
| 10-fold | 97.72 ± 0.88  | 98.26 ± 0.67  | 98.27 ± 0.67  |

*All results are for GLD²-GNN (Ours) as reported in the original paper.*

---

## Project Structure

```
gld2_gnn/
│
├── graph_construction.py   # Predefined 16-node directed graph (S, D matrices)
├── data_loader.py          # PhysioNet loading, gait segmentation, augmentation
├── dgl_unit.py             # 5-branch Dynamic Graph Learning unit
├── dydgn_unit.py           # Dynamic Directed Graph Network unit
├── tcn_unit.py             # Temporal Convolutional Network unit
├── model.py                # Full GLD²-GNN model + two-stream fusion
├── model_improved.py       # GLD²-GNN+ with Adaptive Fusion & Uncertainty
├── train.py                # Training loop, cross-dataset & k-fold CV
├── train_improved.py       # Training script comparing baseline and improved models
├── download_data.py        # PhysioNet dataset downloader
│
├── app.py                  # Streamlit dashboard entry point
├── pages/                  # Streamlit dashboard pages
│   ├── 1_gap_analysis.py   # Gap Analysis
│   ├── 2_live_inference.py # Live Inference
│   ├── 3_graph_viewer.py   # Dynamic Graph Viewer
│   └── 4_results.py        # Results Comparison
│
├── requirements.txt
├── .gitignore
│
├── data/                   # Downloaded datasets (created by download_data.py)
│   ├── Ga/
│   ├── Ju/
│   └── Si/
│
└── checkpoints/            # Saved model weights + result JSON (auto-created)
```

---

## 🛠 Prerequisites & Installation

### Requirements

- Python 3.10+
- GPU strongly recommended (paper uses NVIDIA GeForce RTX 4090). The model will run on CPU but training will be significantly slower.

### Environment Setup

Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/yourusername/gld2-gnn.git
cd gld2-gnn

# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For GPU support (CUDA 12.x):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn scipy numpy requests
```

For CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn scipy numpy requests
```

---

## 💾 Dataset Setup

This project uses the **PhysioNet Gait in Parkinson's Disease (gaitpdb)** database, which contains VGRF recordings from 3 independent study groups.

### Step 1 — Create a PhysioNet account

Register for a free account at [https://physionet.org](https://physionet.org).

### Step 2 — Sign the data use agreement

Visit the dataset page:
[https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/)

Scroll to the bottom and click **"Sign the data use agreement"** while logged in. This step is required — the download will fail without it.

### Step 3 — Download the data

Open `download_data.py` and set your credentials:

```python
USERNAME = "your_physionet_username"
PASSWORD = "your_physionet_password"
```

Then run:

```bash
python download_data.py
```

This authenticates with PhysioNet, retrieves all `.txt` files, and sorts them into the correct local folder structure:

```
data/
  Ga/   GaCo01_01.txt, GaPd01_01.txt, ...   (113 files)
  Ju/   JuCo01_01.txt, JuPd01_01.txt, ...   (129 files)
  Si/   SiCo01_01.txt, SiPd01_01.txt, ...   ( 64 files)
```

### Dataset summary

| Dataset | PD subjects | CO subjects | Total records |
|---------|-------------|-------------|---------------|
| Ga      | 29          | 18          | 113           |
| Ju      | 29          | 25          | 129           |
| Si      | 35          | 29          | 64            |
| **Total** | **93**    | **72**      | **306**       |

Each recording is approximately 2 minutes of walking at comfortable speed, collected with 8 pressure sensors per foot (16 total).

---

## 🚀 Running Experiments

### Sanity-check individual modules

Run each file directly to verify everything is working before training:

```bash
python graph_construction.py   # verifies 16 nodes, 26 edges, S/D matrices
python data_loader.py          # verifies loading, segmentation, augmentation
python dgl_unit.py             # verifies 5-branch adjacency learning
python dydgn_unit.py           # verifies node+edge feature aggregation
python tcn_unit.py             # verifies temporal conv shapes + receptive field
python model.py                # verifies full forward pass + parameter count
```

All should print `All assertions passed.`

### Cross-dataset validation (Table IV)

Train on two datasets, test on the held-out third. Each experiment is repeated 4 times and averaged.

**Ga + Ju → Si:**
```bash
python train.py --mode cross --train_sets Ga Ju --test_set Si \
                --data_root ./data --epochs 120 --batch_size 64
```

**Ga + Si → Ju:**
```bash
python train.py --mode cross --train_sets Ga Si --test_set Ju \
                --data_root ./data --epochs 120 --batch_size 64
```

**Si + Ju → Ga:**
```bash
python train.py --mode cross --train_sets Si Ju --test_set Ga \
                --data_root ./data --epochs 120 --batch_size 64
```

### Mixed-data k-fold cross-validation (Table V)

Merges all three datasets and evaluates with stratified k-fold splitting.

**5-fold:**
```bash
python train.py --mode kfold --k_folds 5 \
                --data_root ./data --epochs 120 --batch_size 64
```

**10-fold:**
```bash
python train.py --mode kfold --k_folds 10 \
                --data_root ./data --epochs 120 --batch_size 64
```

### Enable mixed-precision (faster on RTX GPUs)

Add `--amp` to any command:
```bash
python train.py --mode cross --train_sets Ga Ju --test_set Si \
                --data_root ./data --epochs 120 --batch_size 64 --amp
```

### Full CLI reference

| Argument         | Default         | Description                                        |
|------------------|-----------------|----------------------------------------------------|
| `--mode`         | `cross`         | `cross` = cross-dataset, `kfold` = k-fold CV       |
| `--data_root`    | `./data`        | Path to the downloaded dataset directory           |
| `--train_sets`   | `Ga Ju`         | Datasets for training (cross mode only)            |
| `--test_set`     | `Si`            | Dataset for testing (cross mode only)              |
| `--k_folds`      | `5`             | Number of folds (kfold mode only)                  |
| `--T`            | `128`           | Gait-cycle length — must be divisible by 8         |
| `--aug_factor`   | `10`            | Augmentation multiplier (training data only)       |
| `--epochs`       | `120`           | Maximum training epochs per repeat/fold            |
| `--batch_size`   | `64`            | Mini-batch size                                    |
| `--lr`           | `5e-4`          | Initial learning rate                              |
| `--weight_decay` | `5e-4`          | Adam weight decay                                  |
| `--patience`     | `30`            | Early stopping patience (epochs)                   |
| `--repeats`      | `4`             | Repetitions per cross-dataset experiment           |
| `--seed`         | `42`            | Base random seed                                   |
| `--amp`          | off             | Enable mixed-precision training (RTX GPUs)         |
| `--no_cuda`      | off             | Force CPU-only training                            |
| `--save_dir`     | `./checkpoints` | Output directory for checkpoints and result JSON   |

---

## 📊 Reproducing Paper Results

The following settings exactly match Section IV-A-2 of the paper:

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam, β₁=0.9, β₂=0.995 |
| Weight decay | 5e-4 |
| Learning rate | 5e-4 (initial) |
| LR scheduler | CosineAnnealingLR, T\_max=14, η\_min=1e-5 |
| Loss | Binary cross-entropy |
| Batch size | 64 |
| Max epochs | 120 |
| Early stopping patience | 30 |
| Validation split | 20% of training set |
| Cross-dataset repeats | 4 |

All of these are the defaults in `train.py` — the commands in the section above will use the correct settings without any extra flags.

Results are saved automatically to `./checkpoints/` as JSON files after each run:

```
checkpoints/
  cross_GaJu_Si.json       ← Ga+Ju → Si results (4 repeats)
  cross_GaSi_Ju.json       ← Ga+Si → Ju results (4 repeats)
  cross_SiJu_Ga.json       ← Si+Ju → Ga results (4 repeats)
  kfold5_results.json      ← 5-fold CV results
  kfold10_results.json     ← 10-fold CV results
```

> **Note on variance:** The paper does not specify fixed random seeds for each repeat, so exact numerical reproduction may vary slightly. Mean ± std across 4 repeats should fall within the paper's reported standard deviations.

---

## 🤝 Credits & Citation

### Original Paper

This repository is an independent, ground-up implementation of:

```bibtex
@article{wang2025gld2gnn,
  author    = {Wang, Xiaotian and Zhou, Guanhai and Zhao, Zhifu and
               Zhang, Xiaoyi and Li, Fu and Qi, Fei},
  title     = {A Global-Local Dynamic Directed Graph Neural Network
               for Parkinson's Disease Detection},
  journal   = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume    = {33},
  pages     = {3947--3957},
  year      = {2025},
  doi       = {10.1109/TNSRE.2025.3614430},
  publisher = {IEEE}
}
```

If you use this implementation in your own research, please cite the original paper above.

### Dataset

```bibtex
@misc{physionet_gaitpdb,
  author = {Goldberger, Ary L. and others},
  title  = {Gait in Parkinson's Disease},
  year   = {2000},
  url    = {https://physionet.org/content/gaitpdb/1.0.0/},
  note   = {PhysioNet. Version 1.0.0}
}
```

Original dataset studies:
- **Ga:** Yogev et al., *"Dual tasking, gait rhythmicity, and Parkinson's disease"*, European Journal of Neuroscience, 2005.
- **Ju:** Hausdorff et al., *"Rhythmic auditory stimulation modulates gait variability in Parkinson's disease"*, European Journal of Neuroscience, 2007.
- **Si:** Frenkel-Toledo et al., *"Treadmill walking as an external pacemaker to improve gait rhythm and stability in Parkinson's disease"*, Movement Disorders, 2005.

### Framework & Libraries

This implementation is built on:
- [PyTorch](https://pytorch.org/) — model training and tensor operations
- [scikit-learn](https://scikit-learn.org/) — k-fold splitting and metrics
- [SciPy](https://scipy.org/) — cubic spline interpolation for gait resampling
- [PhysioNet](https://physionet.org/) — dataset hosting and access

---

## ⚠️ Disclaimer

This is an **unofficial implementation** developed for academic coursework and research purposes only.

- This repository is **not affiliated with** the original authors, Xidian University, or IEEE.
- Some architectural details not fully specified in the paper required reasonable design choices — these are documented in the source files.
- This code is **not intended for clinical use**. Parkinson's disease diagnosis must only be performed by qualified medical professionals.
- The PhysioNet dataset requires signing a data use agreement — ensure you comply with all dataset terms before downloading or using the data.
