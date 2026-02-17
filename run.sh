#!/usr/bin/env bash
# Convenience wrapper — runs an experiment inside Docker.
#
# Usage:
#   ./run.sh --experiment 1
#   ./run.sh --experiment 5 --epochs 100 --no-umap
#   ./run.sh --help
#
# To re-generate visualizations:
#   ./run.sh --visualize --experiment 1

set -euo pipefail

if [[ "${1:-}" == "--visualize" ]]; then
    shift
    exec docker compose run --rm visualize "$@"
fi

exec docker compose run --rm experiment "$@"
