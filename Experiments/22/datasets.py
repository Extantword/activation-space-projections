# Experiment 22: Voronoi from Symmetric Seeds (D_n Voronoi Tessellation)
#
# Seed points are placed in a fundamental wedge and replicated by
# the dihedral group D_n.  The Voronoi cell boundaries (where a pixel
# is equidistant from two seeds) inherit the same D_n symmetry.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N Voronoi tessellation images with D_n symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} symmetric Voronoi images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)

    for i in range(n):
        cx, cy = size / 2, size / 2
        fold = random.choice([3, 4, 5, 6, 8])
        num_base = random.randint(3, 8)

        # Place seeds in one wedge, then replicate with D_n
        seeds = []
        for _ in range(num_base):
            r = random.uniform(3, size / 2 - 2)
            a = random.uniform(0, math.pi / fold)
            for k in range(fold):
                offset = k * 2 * math.pi / fold
                for mirror in [1, -1]:
                    angle = mirror * a + offset
                    sx = cx + r * math.cos(angle)
                    sy = cy + r * math.sin(angle)
                    seeds.append((sx, sy))

        seeds_arr = np.array(seeds)  # (num_seeds, 2)

        # Compute nearest seed for every pixel
        dists = ((x_grid[:, :, np.newaxis] - seeds_arr[np.newaxis, np.newaxis, :, 0]) ** 2 +
                 (y_grid[:, :, np.newaxis] - seeds_arr[np.newaxis, np.newaxis, :, 1]) ** 2)
        nearest = np.argmin(dists, axis=-1)

        # Extract boundaries (pixels where a neighbour has a different seed)
        boundary = np.zeros((size, size), dtype=np.uint8)
        diff_h = nearest[:, :-1] != nearest[:, 1:]
        diff_v = nearest[:-1, :] != nearest[1:, :]
        boundary[:, :-1] |= (diff_h * 255).astype(np.uint8)
        boundary[:-1, :] |= (diff_v * 255).astype(np.uint8)

        images[i] = boundary

    print("Generation complete.")
    return images, labels
