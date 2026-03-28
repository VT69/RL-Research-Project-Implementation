# GLD²-GNN: Gait Graph Learning for Parkinson's Disease Detection

An implementation of the published research paper on graph learning for Parkinson's disease detection using Vertical Ground Reaction Force (VGRF) data.

This repository implements **GLD²-GNN** (Spatial-Temporal and Dynamic Graph Neural Networks) designed to identify Parkinson's Disease directly from gait data cycles extracted from foot sensors. 

It provides standard routines for data downloading, segmentation, augmentation, spatial-temporal graph construction, two-stream graph parsing, and training via cross-dataset or mixed-data K-fold cross-validation.

---

## 🛠 Prerequisites & Installation

### 1. Requirements
Ensure you are running Python 3.8+ before setting up the dependencies. It's highly recommended to use a virtual environment or `conda` environment.
Hardware requirements: GPU (Nvidia RTX series recommended) for accelerated training. The model will run on CPU, but training will take significantly longer.

### 2. Environment Setup

Clone this repository and open your terminal. From the root of this project:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# -> On Windows:
venv\Scripts\activate
# -> On Linux / macOS:
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

---

## 💾 Dataset Preparation

This project uses the **PhysioNet Gait in Parkinson's Disease (gaitpdb)** dataset. The dataset includes 3 independent sub-datasets (`Ga`, `Ju`, `Si`) containing VGRF records.

> **Important**: This dataset is credentialed. To download it, you must register on [PhysioNet](https://physionet.org), agree to their terms, and sign the Data Use Agreement for this specific dataset here:
> [Gait in Parkinson's Disease Dataset](https://physionet.org/content/gaitpdb/1.0.0/)

Once access is granted, open `data_download_script.py` and input your PhysioNet `USERNAME` and `PASSWORD`:
```python
USERNAME = "your_username"
PASSWORD = "your_password"
```

Next, run the downloader script. This will automatically authenticate, retrieve the necessary text files, and sort them into the correct folder structures required by the model.

```bash
python data_download_script.py
```
After successful execution, your data must reside inside a top-level `data/` directory like so:
```
data/
 ├── Ga/
 │    └── GaCo01_01.txt, GaPt03_01.txt, etc.
 ├── Ju/
 │    └── JuCo01_01.txt, JuPt01_01.txt, etc.
 └── Si/
      └── SiCo01_01.txt, SiPt02_01.txt, etc.
```

---

## 🚀 Training & Evaluation Pipeline

The training framework `train.py` supports two separate evaluation methods natively described in the paper: **Cross-Dataset Validation** and **K-Fold Mixed-Data Cross-Validation**. The framework will automatically handle data loading, segmentation, fixed-length interpolation (via cubic splines), building the motion graph streams, and data augmentations (permutation and window slicing).

### 1. Cross-Dataset Validation
Train the model on independent specific subsets, and test on the remaining unseen subset.

**Example**: Train on `Ga` and `Ju`, Test on `Si`.
```bash
python train.py --data_root ./data --mode cross --train_sets Ga Ju --test_set Si --batch_size 64 --epochs 120 --amp
```
*(Optionally include the `--amp` flag to enable mixed precision training for RTX hardware).*

### 2. K-Fold Cross Validation
Train using an $N$-Fold cross-validation split merging all 3 subsets (`Ga`, `Ju`, `Si`). 

**Example**: 5-Fold stratified cross-validation on all merged data.
```bash
python train.py --data_root ./data --mode kfold --k_folds 5 --batch_size 64 --epochs 120 --amp
```

### Full Configuration Reference
Check out the CLI args to configure sequence length, epochs, batches, and patience:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | string | `cross` | Defines CV method: `cross` or `kfold` |
| `--data_root` | string | `./data` | Directory containing the PhysioNet dataset folders |
| `--train_sets` | list | `Ga Ju` | Target datasets for training (Cross Mode) |
| `--test_set`| string | `Si` | Target dataset for validation (Cross Mode) |
| `--k_folds` | int | `5` | Total data splits to evaluate (KFold Mode) |
| `--epochs` | int | `120` | Max number of epochs to run per fold/repeat |
| `--T` | int | `128` | Gait-cycle sequence duration to construct |
| `--batch_size` | int | `64` | Target DataLoader Mini-batch configuration |
| `--amp` | flag | N/A | Enable pytorch automatic mixed-precision training |
| `--no_cuda` | flag | N/A | Explicitly specify CPU-only training computation |

---

## 📊 Results Summary

The architecture evaluates with accuracy, F1 score and Geometric Mean metrics.
Upon running training, checkpoints mapped by run configuration will be saved out automatically via validation loss monitoring. Output JSON metadata reports featuring performance arrays across metrics will be packaged into the default `./checkpoints` directory, to safely extract graphs and comparative benchmarks over repetitions.

## 🤝 Credits & Acknowledgements
This is an unofficial implementation of the GLD²-GNN method for Parkinson's Disease detection using VGRF signals. Please refer to and cite the original literature and authors if utilizing this architecture in your own extended research. The framework extensively relies on PyTorch and standard implementations of TCN and GNN units.

Data is provided by PhysioNet (gaitpdb dataset). All dataset terms of use apply to downstream consumers.
