# Local Setup Guide

## Quick Start (Docker — recommended)

Only requirement on the server: **Docker** (and **Docker Compose**).

```bash
git clone <repo-url>
cd activation-space-projections

# 1. One-time setup — detects GPU, builds the Docker image
./setup.sh

# 2. Run an experiment
N_EXPERIMENT=1 ./run.sh
```

That's it. No Python, no pip, no virtual environments.

### Environment variables

Configure experiments with these env vars:

| Variable | Required | Description |
|----------|----------|-------------|
| `N_EXPERIMENT` | yes | Experiment ID (1–45) |
| `N_EPOCHS` | no | Number of training epochs (default 50) |
| `LATENT_DIM` | no | Autoencoder bottleneck dimension (default 64) |
| `IMAGE_SIZE` | no | Image resolution (default 64) |

```bash
# Custom parameters
N_EXPERIMENT=5 N_EPOCHS=100 LATENT_DIM=128 IMAGE_SIZE=64 ./run.sh

# All 45 experiments
for i in $(seq 1 45); do N_EXPERIMENT=$i ./run.sh; done

# Re-generate visualizations from saved data
N_EXPERIMENT=1 ./run.sh --visualize

# Extra flags are passed through directly
N_EXPERIMENT=1 ./run.sh --no-umap --seed 42
```

### GPU support

`./setup.sh` auto-detects NVIDIA GPUs. If one is found and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
is installed, the image is built with CUDA support automatically.

To force CPU or GPU mode:

```bash
./setup.sh --cpu    # force CPU even if GPU is present
./setup.sh --gpu    # force GPU
```

To revert from GPU to CPU later:

```bash
rm docker-compose.override.yml && docker compose build
```

### GitHub Pages site

The site is **deployed automatically** via GitHub Actions. Whenever you
push changes to `outputs/` or `Experiments/` on `main`, the workflow
rebuilds the site and deploys it to GitHub Pages — no manual steps needed.

**One-time setup:** go to **Settings > Pages > Source** and select
**GitHub Actions** (instead of "Deploy from a branch").

You can also trigger a deploy manually from the **Actions** tab
(`workflow_dispatch`).

To preview the site locally before pushing:

```bash
./run.sh --site          # builds into docs/
open docs/index.html     # preview in browser
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
├── setup.sh                     # One-time setup: detects GPU, builds Docker image
├── run.sh                       # Run experiments: N_EXPERIMENT=1 ./run.sh
├── build_site.py                # Generate GitHub Pages site from outputs
├── .github/workflows/           # GitHub Actions (auto-deploy site on push)
├── Dockerfile                   # Docker image (CPU by default, GPU via build arg)
├── docker-compose.yml           # docker compose services
├── .dockerignore
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
