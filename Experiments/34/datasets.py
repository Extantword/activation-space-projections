# Experiment 34: Symmetric Contour Maps (D_n Iso-Lines)
#
# Gaussian blobs are placed with D_n symmetry (like Experiment 24) and
# their sum is computed as a smooth field.  Instead of displaying the
# raw field, iso-contour lines at evenly-spaced thresholds are drawn,
# producing a topographic-map pattern with D_n symmetry.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} symmetric contour-map images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2

    for i in range(n):
        fold = random.choice([3, 4, 5, 6])
        num_base = random.randint(2, 5)
        sigma = random.uniform(size * 0.08, size * 0.18)

        field = np.zeros((size, size), dtype=np.float64)

        for _ in range(num_base):
            r = random.uniform(size * 0.1, size * 0.35)
            a = random.uniform(0, math.pi / fold)
            amp = random.uniform(0.5, 1.0)
            for k in range(fold):
                for mirror in [1, -1]:
                    angle = mirror * a + k * 2 * math.pi / fold
                    bx = cx + r * math.cos(angle)
                    by = cy + r * math.sin(angle)
                    field += amp * np.exp(
                        -((x_grid - bx) ** 2 + (y_grid - by) ** 2)
                        / (2 * sigma ** 2)
                    )

        # Draw contour lines at evenly-spaced thresholds
        if field.max() > 0:
            field /= field.max()  # normalise to [0, 1]

        num_levels = random.randint(5, 10)
        width = 0.02
        canvas = np.zeros((size, size), dtype=np.float64)
        for lev in range(1, num_levels + 1):
            threshold = lev / (num_levels + 1)
            canvas += np.exp(-((field - threshold) / width) ** 2)

        if canvas.max() > 0:
            canvas = canvas / canvas.max() * 255
        images[i] = canvas.astype(np.uint8)

    print("Generation complete.")
    return images, labels
