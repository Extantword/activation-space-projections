# =============================================================
# activation-space-projections
#
# Build (CPU, default):
#   docker build -t asp .
#
# Build (GPU — requires nvidia-container-toolkit on host):
#   docker build --build-arg DEVICE=gpu -t asp-gpu .
# =============================================================

ARG DEVICE=cpu

# ---- CPU base ----
FROM python:3.10-slim AS base-cpu

# ---- GPU base (CUDA 11.8) ----
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# ---- Final stage ----
FROM base-${DEVICE} AS final

WORKDIR /workspace

# Install PyTorch first (correct index for CPU vs CUDA), then the rest
ARG DEVICE=cpu
COPY requirements.txt .
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118 ; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
CMD ["run_experiment.py", "--help"]
