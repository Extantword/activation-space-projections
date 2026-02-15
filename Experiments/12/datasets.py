# Experiment 12: Chladni Figures (Vibrating-Plate Nodal Lines)
#
# The nodal lines of a vibrating square plate are approximated by:
#   f(x,y) = cos(m*pi*x)*cos(n*pi*y) +/- cos(n*pi*x)*cos(m*pi*y)
# Pixels near f=0 are highlighted, producing the characteristic sand
# patterns seen in Ernst Chladni's experiments.

import numpy as np
import random


def generate_dataset(n=5000, size=64):
    """
    Generates N Chladni-figure images with reflective symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} Chladni figure images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    # Normalised coordinate grids [0, 1]
    y_norm, x_norm = np.mgrid[0:size, 0:size].astype(np.float64) / size

    for i in range(n):
        m = random.randint(1, 8)
        n_mode = random.randint(1, 8)
        sign = random.choice([1, -1])

        f = (np.cos(m * np.pi * x_norm) * np.cos(n_mode * np.pi * y_norm)
             + sign * np.cos(n_mode * np.pi * x_norm) * np.cos(m * np.pi * y_norm))

        # Highlight the nodal lines (where f is near zero)
        width = 0.15
        nodal = np.exp(-(f / width) ** 2) * 255
        images[i] = nodal.astype(np.uint8)

    print("Generation complete.")
    return images, labels
