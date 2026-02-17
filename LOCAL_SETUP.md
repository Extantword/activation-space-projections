# Local Setup Guide

## Quick Start (Docker — recommended)

Only requirement on the server: **Docker** (and **Docker Compose**).

```bash
git clone <repo-url>
cd activation-space-projections

# Run experiment 1 — builds the image automatically on first run
./run.sh --experiment 1

# Or equivalently:
docker compose run --rm experiment --experiment 1
```

That's it. No Python, no pip, no virtual environments.

### More examples

```bash
# Custom parameters
./run.sh --experiment 5 --epochs 100 --latent-dim 128

# All 45 experiments
for i in $(seq 1 45); do ./run.sh --experiment $i; done

# Re-generate visualizations from saved data
./run.sh --visualize --experiment 1
```

### GPU support

If the server has an NVIDIA GPU, install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
then uncomment the `gpu-experiment` service in `docker-compose.yml` and run:

```bash
docker compose run --rm gpu-experiment --experiment 1
```

### Outputs

Results are written to the `outputs/` directory on the host (mounted as
a Docker volume), so they persist across runs.

---

## Manual Setup (without Docker)

### Prerequisites

- **Python 3.10** (the version the notebooks were developed on)
- **pip** (Python package manager)
- **(Optional) NVIDIA GPU** with CUDA drivers for accelerated training. The code falls back to CPU automatically if no GPU is available.

### Installation

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

#### PyTorch with GPU (CUDA) support

The `requirements.txt` installs the default (CPU) PyTorch build. If the server has an NVIDIA GPU, install the CUDA-enabled build instead:

```bash
# Example for CUDA 11.8 – check https://pytorch.org/get-started/locally/ for your version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Running Experiments

#### Option A: Command-line scripts (recommended for servers)

No Jupyter needed. Run everything from the terminal:

```bash
# Run a single experiment with default settings
python run_experiment.py --experiment 1

# Customise parameters
python run_experiment.py --experiment 5 \
    --epochs 100 \
    --latent-dim 128 \
    --image-size 64 \
    --n-samples 20000

# Skip UMAP for faster runs
python run_experiment.py --experiment 1 --no-umap

# Set a seed for reproducibility
python run_experiment.py --experiment 1 --seed 42

# Run all 45 experiments in sequence
for i in $(seq 1 45); do
    python run_experiment.py --experiment $i
done
```

Re-generate visualizations without re-training:

```bash
python visualize.py --experiment 1
python visualize.py --data-dir outputs/experiment_5
```

Full CLI flags for `run_experiment.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment`, `-e` | *(required)* | Experiment ID (1-45) |
| `--image-size` | 64 | Image resolution |
| `--n-samples` | 10000 | Number of images to generate |
| `--latent-dim` | 64 | Autoencoder bottleneck dimension |
| `--batch-size` | 64 | Training batch size |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--dropout` | 0.05 | Dropout rate |
| `--output-dir` | `outputs/experiment_<ID>` | Output directory |
| `--no-umap` | off | Skip UMAP (faster) |
| `--seed` | none | Random seed |

#### Option B: Jupyter notebooks

```bash
jupyter notebook
```

Then open:

1. **`experiment_runner.ipynb`** — runs a single experiment interactively.
2. **`visualize_results.ipynb`** — loads saved results and renders interactive Plotly visualizations.

#### Note on Colab-specific code

`curves.py` contains a `google.colab.output` import. This is wrapped in a try/except and will be silently skipped when running locally — no action needed.

## Outputs

Each experiment writes to `outputs/experiment_<ID>/`:

| File | Description |
|------|-------------|
| `samples.png` | 10 random dataset samples |
| `loss_curve.png` | Training & validation loss over epochs |
| `reconstructions.png` | Original vs. reconstructed images |
| `pca_3d.html` | Interactive 3D PCA scatter (open in browser) |
| `umap_3d.html` | Interactive 3D UMAP scatter (open in browser) |
| `latent_data.npz` | Raw data for `visualize.py` re-use |

## Project Structure

```
activation-space-projections/
├── Dockerfile                   # Docker image (CPU by default, GPU via build arg)
├── docker-compose.yml           # docker compose services
├── .dockerignore
├── run.sh                       # Convenience wrapper: ./run.sh --experiment 1
├── requirements.txt             # Python dependencies
├── LOCAL_SETUP.md               # This file
├── run_experiment.py            # CLI script — train & project
├── visualize.py                 # CLI script — regenerate HTML plots
├── plotting.py                  # Shared Plotly 3D figure builder
├── experiment_runner.ipynb      # Original Colab notebook
├── visualize_results.ipynb      # Original Colab notebook
├── curves.py                    # Shared utilities (used by notebooks)
└── Experiments/
    ├── 1/datasets.py            # Dataset generator for experiment 1
    ├── 2/datasets.py
    ├── ...
    └── 45/datasets.py
```
