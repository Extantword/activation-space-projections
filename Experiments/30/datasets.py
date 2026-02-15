# Experiment 30: Damped Radial Harmonics (Decaying Angular Modes)
#
# A radial pattern combining an angular cosine mode with a damped
# radial sinusoid:
#   f(r, theta) = cos(n*theta) * exp(-r/sigma) * sin(freq*r)
# The exponential decay confines the pattern near the centre, while
# the angular mode gives C_n or D_n symmetry.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N damped radial-harmonic images.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} damped radial harmonic images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    theta_grid = np.arctan2(y_grid - cy, x_grid - cx)

    for i in range(n):
        n_mode = random.randint(2, 10)
        freq = random.uniform(0.3, 1.0)
        sigma = random.uniform(size * 0.2, size * 0.5)
        phase = random.uniform(0, 2 * math.pi)

        angular = np.cos(n_mode * theta_grid + phase)
        radial = np.sin(freq * r_grid)
        decay = np.exp(-r_grid / sigma)

        pattern = angular * radial * decay

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
