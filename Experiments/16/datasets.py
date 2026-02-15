# Experiment 16: Circular Membrane Harmonics (Drum Modes)
#
# Standing-wave patterns on a circular drum head, approximated by:
#   f(r, theta) = cos(n * theta) * sin(freq * r)
# where (r, theta) are polar coordinates from the image centre.
# The angular mode number n gives C_n rotational symmetry, while
# the radial frequency controls the number of concentric nodal rings.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N circular-membrane harmonic images.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} drum-mode images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    theta_grid = np.arctan2(y_grid - cy, x_grid - cx)
    max_r = size / 2

    for i in range(n):
        n_mode = random.randint(1, 8)       # angular mode
        freq = random.uniform(0.15, 0.5)    # radial frequency
        phase = random.uniform(0, 2 * math.pi)

        angular = np.cos(n_mode * theta_grid + phase)
        radial = np.sin(freq * r_grid * math.pi)

        # Circular mask (fade to zero at boundary)
        envelope = np.clip(1.0 - r_grid / max_r, 0, 1)

        pattern = angular * radial * envelope
        # Rescale to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
