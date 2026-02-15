# Experiment 9: Moire Interference (Bilateral Symmetry from Concentric Circles)
#
# Two sets of concentric-circle gratings whose centres are placed
# symmetrically about the image centre.  The resulting interference
# (Moire) fringes inherit bilateral symmetry along the axis connecting
# the two centres.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N Moire interference images with bilateral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} Moire images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    # Pre-compute coordinate grid (centred)
    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)

    for i in range(n):
        # Two centres placed symmetrically about the image centre
        offset = random.uniform(3, size / 4)
        axis_angle = random.uniform(0, 2 * math.pi)
        cx1 = size / 2 + offset * math.cos(axis_angle)
        cy1 = size / 2 + offset * math.sin(axis_angle)
        cx2 = size / 2 - offset * math.cos(axis_angle)
        cy2 = size / 2 - offset * math.sin(axis_angle)

        freq = random.uniform(0.5, 1.5)

        r1 = np.sqrt((x_grid - cx1) ** 2 + (y_grid - cy1) ** 2)
        r2 = np.sqrt((x_grid - cx2) ** 2 + (y_grid - cy2) ** 2)

        g1 = np.sin(freq * r1)
        g2 = np.sin(freq * r2)

        pattern = (g1 + g2) / 2  # range [-1, 1]
        pattern = ((pattern + 1) / 2 * 255).astype(np.uint8)

        images[i] = pattern

    print("Generation complete.")
    return images, labels
