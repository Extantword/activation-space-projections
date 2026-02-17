# Local Setup Guide

## Prerequisites

- **Python 3.10** (the version the notebooks were developed on)
- **pip** (Python package manager)
- **(Optional) NVIDIA GPU** with CUDA drivers for accelerated training. The code falls back to CPU automatically if no GPU is available.

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd activation-space-projections

# 2. Create a virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch with GPU (CUDA) support

The `requirements.txt` installs the default (CPU) PyTorch build. If the server has an NVIDIA GPU, install the CUDA-enabled build instead:

```bash
# Example for CUDA 11.8 – check https://pytorch.org/get-started/locally/ for your version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Running the Notebooks

```bash
# Start Jupyter
jupyter notebook
```

Then open:

1. **`experiment_runner.ipynb`** — runs all 45 experiments (training, activation extraction, dimensionality reduction).
2. **`visualize_results.ipynb`** — loads saved results and produces interactive Plotly visualizations.

### Note on Colab-specific code

`curves.py` contains a `google.colab.output` import. This is wrapped in a try/except and will be silently skipped when running locally — no action needed.

## Project Structure

```
activation-space-projections/
├── requirements.txt             # Python dependencies
├── LOCAL_SETUP.md               # This file
├── experiment_runner.ipynb      # Main experiment notebook
├── visualize_results.ipynb      # Visualization notebook
├── curves.py                    # Shared utilities (training, plotting, UMAP, etc.)
└── Experiments/
    ├── 1/datasets.py            # Dataset generator for experiment 1
    ├── 2/datasets.py
    ├── ...
    └── 45/datasets.py
```
