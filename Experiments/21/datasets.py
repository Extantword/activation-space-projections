# Experiment 21: Polar Modulated Gratings (Angular-Radial Interference)
#
# Intensity is computed in polar coordinates as the product of an
# angular modulation and a radial oscillation:
#   f(r, theta) = sin(n*theta + freq*r)
# This creates spiral-like interference fringes with approximate
# n-fold rotational symmetry.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N polar-modulated grating images.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} polar grating images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    theta_grid = np.arctan2(y_grid - cy, x_grid - cx)

    for i in range(n):
        n_fold = random.choice([3, 4, 5, 6, 8, 10])
        freq = random.uniform(0.2, 0.8)
        phase = random.uniform(0, 2 * math.pi)

        pattern = np.sin(n_fold * theta_grid + freq * r_grid + phase)
        pattern = ((pattern + 1) / 2 * 255).astype(np.uint8)
        images[i] = pattern

    print("Generation complete.")
    return images, labels
