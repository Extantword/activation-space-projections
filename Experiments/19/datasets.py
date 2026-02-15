# Experiment 19: Wave Interference Ring (N-Source Circular Array)
#
# N point sources are placed equally spaced on a circle centred in the
# image.  Each source emits circular sinusoidal waves, and the pixel
# intensity is the sum of all wave amplitudes.  The N-fold rotational
# symmetry of the source layout is inherited by the interference pattern.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N wave-interference images from a ring of sources.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} wave-interference images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)

    for i in range(n):
        cx, cy = size / 2, size / 2
        num_sources = random.choice([3, 4, 5, 6, 8])
        ring_radius = random.uniform(size * 0.15, size * 0.35)
        freq = random.uniform(0.5, 1.5)

        pattern = np.zeros((size, size), dtype=np.float64)

        for k in range(num_sources):
            angle = 2 * math.pi * k / num_sources
            sx = cx + ring_radius * math.cos(angle)
            sy = cy + ring_radius * math.sin(angle)
            r = np.sqrt((x_grid - sx) ** 2 + (y_grid - sy) ** 2)
            pattern += np.sin(freq * r)

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
