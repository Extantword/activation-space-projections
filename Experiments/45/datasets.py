# Experiment 45: Superposed Harmonic Modes (Multi-Modal Wave Field)
#
# A sum of 2-5 randomly-chosen angular-radial harmonic terms:
#   f(r, theta) = sum_k  A_k * cos(n_k * theta + phi_k) * sin(freq_k * r)
# Each term is like a single drum mode (Experiment 16), but the
# superposition of multiple modes at different angular orders and
# radial frequencies creates rich, complex wave fields that combine
# the symmetries of the individual modes.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} superposed harmonic images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    theta_grid = np.arctan2(y_grid - cy, x_grid - cx)
    max_r = size / 2

    for i in range(n):
        num_modes = random.randint(2, 5)
        pattern = np.zeros((size, size), dtype=np.float64)

        for _ in range(num_modes):
            n_mode = random.randint(0, 8)
            freq = random.uniform(0.15, 0.8)
            phase = random.uniform(0, 2 * math.pi)
            amp = random.uniform(0.3, 1.0)

            angular = np.cos(n_mode * theta_grid + phase)
            radial = np.sin(freq * r_grid * math.pi)
            pattern += amp * angular * radial

        # Circular envelope
        envelope = np.clip(1.0 - r_grid / max_r, 0, 1)
        pattern *= envelope

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
