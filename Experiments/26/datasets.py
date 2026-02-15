# Experiment 26: Interference Lattice (Plane-Wave Superposition)
#
# N plane waves propagating at equally-spaced angles are summed:
#   f(x,y) = sum_k  sin(freq * (x*cos(angle_k) + y*sin(angle_k)))
# The resulting pattern has N-fold rotational symmetry and resembles
# 2-D crystallographic lattice / wallpaper patterns.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N interference-lattice images with N-fold symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} interference lattice images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    x_c = x_grid - size / 2
    y_c = y_grid - size / 2

    for i in range(n):
        num_waves = random.choice([3, 4, 5, 6])
        freq = random.uniform(0.3, 0.8)
        base_angle = random.uniform(0, 2 * math.pi)

        pattern = np.zeros((size, size), dtype=np.float64)
        for k in range(num_waves):
            angle = base_angle + 2 * math.pi * k / num_waves
            pattern += np.sin(freq * (x_c * math.cos(angle)
                                      + y_c * math.sin(angle)))

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
