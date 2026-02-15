# Experiment 24: Symmetric Gaussian Blobs (Bilateral Soft Patterns)
#
# Random 2-D Gaussian blobs are placed on the right half of the image
# and mirrored across the vertical axis.  The result is a soft,
# cloud-like image with exact bilateral symmetry — a continuous-tone
# analogue of the Rorschach inkblot (Experiment 7).

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N symmetric Gaussian-blob images with bilateral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} Gaussian-blob images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = size / 2

    for i in range(n):
        num_blobs = random.randint(4, 12)
        canvas = np.zeros((size, size), dtype=np.float64)

        for _ in range(num_blobs):
            # Blob centre on right half
            bx = random.uniform(cx, size - 3)
            by = random.uniform(3, size - 3)
            sigma = random.uniform(3, 10)
            amplitude = random.uniform(0.5, 1.0)

            # Right blob
            g = amplitude * np.exp(
                -((x_grid - bx) ** 2 + (y_grid - by) ** 2)
                / (2 * sigma ** 2)
            )
            # Mirrored (left) blob
            g_mirror = amplitude * np.exp(
                -((x_grid - (2 * cx - bx)) ** 2 + (y_grid - by) ** 2)
                / (2 * sigma ** 2)
            )
            canvas += g + g_mirror

        # Normalise to [0, 255]
        if canvas.max() > 0:
            canvas = canvas / canvas.max() * 255
        images[i] = canvas.astype(np.uint8)

    print("Generation complete.")
    return images, labels
