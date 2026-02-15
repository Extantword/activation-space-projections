# Experiment 41: Rectangular Quantum Modes (Particle-in-a-Box)
#
# Probability density  |psi_{m,n}|^2 = sin^2(m*pi*x) * sin^2(n*pi*y)
# of a quantum particle confined to a 2D square well.  The result is
# a smooth, grid-like intensity field with D2 or D4 symmetry depending
# on whether m equals n.

import numpy as np
import random


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} quantum-mode images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    # Normalised coordinates [0, 1]
    y_norm, x_norm = np.mgrid[0:size, 0:size].astype(np.float64) / size

    for i in range(n):
        m = random.randint(1, 8)
        n_mode = random.randint(1, 8)

        psi_sq = np.sin(m * np.pi * x_norm) ** 2 \
               * np.sin(n_mode * np.pi * y_norm) ** 2

        # Normalise to [0, 255]
        images[i] = (psi_sq * 255).astype(np.uint8)

    print("Generation complete.")
    return images, labels
