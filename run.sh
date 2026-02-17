#!/usr/bin/env bash
# =============================================================
# run.sh — Run experiments or visualizations inside Docker.
#
# Configure via environment variables:
#   N_EXPERIMENT  (required*)  Experiment ID (1–45)
#   N_EPOCHS                   Number of training epochs
#   LATENT_DIM                 Autoencoder bottleneck dimension
#   IMAGE_SIZE                 Image resolution
#
# *N_EXPERIMENT is not required for --site.
#
# Examples:
#   N_EXPERIMENT=1 ./run.sh
#   N_EXPERIMENT=5 N_EPOCHS=100 LATENT_DIM=128 IMAGE_SIZE=64 ./run.sh
#
#   # Re-generate visualizations
#   N_EXPERIMENT=1 ./run.sh --visualize
#
#   # Build the GitHub Pages site from experiment outputs
#   ./run.sh --site
#
#   # Pass extra flags directly
#   N_EXPERIMENT=1 ./run.sh --no-umap --seed 42
# =============================================================
set -euo pipefail

# ── Parse mode ───────────────────────────────────────────────
SERVICE=experiment
EXTRA=()

for arg in "$@"; do
    case "$arg" in
        --visualize) SERVICE=visualize ;;
        --site)      SERVICE=build-site ;;
        *)           EXTRA+=("$arg") ;;
    esac
done

# ── Build site (no experiment needed) ────────────────────────
if [[ "$SERVICE" == "build-site" ]]; then
    exec docker compose run --rm build-site "${EXTRA[@]}"
fi

# ── Build CLI args from env vars ─────────────────────────────
ARGS=()

if [[ -z "${N_EXPERIMENT:-}" ]]; then
    echo "ERROR: N_EXPERIMENT is required."
    echo "Usage: N_EXPERIMENT=1 ./run.sh"
    exit 1
fi
ARGS+=(--experiment "$N_EXPERIMENT")

if [[ "$SERVICE" == "experiment" ]]; then
    [[ -n "${N_EPOCHS:-}" ]]   && ARGS+=(--epochs "$N_EPOCHS")
    [[ -n "${LATENT_DIM:-}" ]] && ARGS+=(--latent-dim "$LATENT_DIM")
    [[ -n "${IMAGE_SIZE:-}" ]] && ARGS+=(--image-size "$IMAGE_SIZE")
fi

# ── Run ──────────────────────────────────────────────────────
exec docker compose run --rm "$SERVICE" "${ARGS[@]}" "${EXTRA[@]}"
