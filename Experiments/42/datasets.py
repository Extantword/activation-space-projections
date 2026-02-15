# Experiment 42: Zernike Polynomial Modes (Optical Aberrations)
#
# Zernike polynomials Z_n^m(r, theta) = R_n^m(r) * cos(m*theta) form
# the standard orthogonal basis on the unit disk.  They describe
# optical wavefront aberrations: defocus, astigmatism, coma, trefoil,
# spherical aberration, etc.  Each image shows a single Zernike mode
# on a circular aperture with smooth intensity gradients.

import numpy as np
import random
import math


def _zernike_radial(n, m, rho):
    """Compute the radial Zernike polynomial R_n^m(rho)."""
    result = np.zeros_like(rho)
    abs_m = abs(m)
    for k in range((n - abs_m) // 2 + 1):
        num = ((-1) ** k) * math.factorial(n - k)
        den = (math.factorial(k)
               * math.factorial((n + abs_m) // 2 - k)
               * math.factorial((n - abs_m) // 2 - k))
        result += (num / den) * rho ** (n - 2 * k)
    return result


# Pre-compute valid (n, m) pairs up to order 8
_VALID_MODES = []
for _n in range(0, 9):
    for _m in range(-_n, _n + 1, 2):
        _VALID_MODES.append((_n, _m))


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} Zernike mode images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    max_r = size / 2
    rho = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / max_r
    theta = np.arctan2(y_grid - cy, x_grid - cx)
    disk_mask = rho <= 1.0

    for i in range(n):
        zn, zm = random.choice(_VALID_MODES)

        R = _zernike_radial(zn, zm, np.clip(rho, 0, 1))
        if zm >= 0:
            angular = np.cos(abs(zm) * theta)
        else:
            angular = np.sin(abs(zm) * theta)

        pattern = R * angular * disk_mask

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
