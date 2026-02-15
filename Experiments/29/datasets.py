# Experiment 29: Pinwheel Sectors (Rotational Spiral Symmetry)
#
# The image is divided into 2*N alternating bright/dark curved sectors
# that spiral outward from the centre.  The curvature comes from
# rotating the sector boundary angle as a function of radius.
# The result has C_N rotational symmetry.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N pinwheel-sector images with C_N rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} pinwheel images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    theta_grid = np.arctan2(y_grid - cy, x_grid - cx)

    for i in range(n):
        num_blades = random.choice([3, 4, 5, 6, 8])
        twist = random.uniform(0.05, 0.2)  # radians per pixel of radius
        phase = random.uniform(0, 2 * math.pi)

        # Effective angle with radial twist
        theta_eff = theta_grid + twist * r_grid + phase

        # Map to sectors
        sector_val = np.sin(num_blades * theta_eff)
        # Convert to binary-ish with soft edges
        pattern = ((np.tanh(sector_val * 5) + 1) / 2 * 255).astype(np.uint8)

        # Circular fade
        fade = np.clip(1.0 - r_grid / (size / 2), 0, 1)
        pattern = (pattern * fade).astype(np.uint8)

        images[i] = pattern

    print("Generation complete.")
    return images, labels
