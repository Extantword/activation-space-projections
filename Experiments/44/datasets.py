# Experiment 44: Double-Slit Interference (Coherent Two-Source Fringes)
#
# Two coherent point sources are placed symmetrically about the image
# centre (along the vertical axis).  The intensity at each pixel is
# the squared magnitude of the sum of two circular waves:
#   I(x,y) = |exp(i*k*r1) + exp(i*k*r2)|^2
#           = 2 + 2*cos(k*(r1 - r2))
# This produces the classic hyperbolic interference fringes with
# bilateral symmetry about both axes.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} double-slit images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2

    for i in range(n):
        # Slit separation and wavenumber
        slit_sep = random.uniform(size * 0.05, size * 0.3)
        k = random.uniform(0.5, 2.0)

        # Two sources along vertical axis, symmetric about centre
        s1y = cy - slit_sep / 2
        s2y = cy + slit_sep / 2

        r1 = np.sqrt((x_grid - cx) ** 2 + (y_grid - s1y) ** 2)
        r2 = np.sqrt((x_grid - cx) ** 2 + (y_grid - s2y) ** 2)

        # Interference intensity: 2 + 2*cos(k*(r1 - r2))
        pattern = 2.0 + 2.0 * np.cos(k * (r1 - r2))

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
