## Overview

**PointClouds** is a lightweight research framework for training and evaluating neural networks on point-cloud data.

It currently provides two main experiment scripts:
- `run_egnn.py` — runs experiments using **EGNN (Equivariant Graph Neural Networks)**
- `run_gramnet.py` — runs experiments using **GramNet**, a bilipschitz-constrained model for stable geometric learning

Both scripts share a unified command-line interface and support a configurable parameter:
- `--noise_level` — controls the amount of Gaussian noise added to the input point clouds (useful for robustness studies)

### Key Features
- 🧠 Clean and minimal Python implementation (no external framework wrappers)
- 📊 Ready-to-run scripts for EGNN and GramNet
- ⚙️ Easily configurable noise level and dataset options
- 💻 Compatible with both CPU and CUDA environments
- 🔍 Designed for clarity and research reproducibility
## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/yonatansverdlov/PointClouds.git
cd PointClouds
conda env create -f env.yaml
conda activate pointclouds
```

## Run EGNN and GramNet experiments

# Choose noise_level 
```bash
## 1) Run EGNN experiment with 5% noise
python run_egnn.py --noise_level noise_level
```
```bash
## 2) Run GramNet experiment with 5% noise
python run_gramnet.py --noise_level noise_level
```
