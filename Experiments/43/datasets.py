# Experiment 43: Beating Radial Waves (Amplitude-Modulated Ripples)
#
# Two circular waves with slightly different frequencies emanate from
# the image centre.  Their sum  sin(f1*r) + sin(f2*r)  equals
# 2*cos(df*r/2)*sin(fc*r/2)  where df = f1-f2 and fc = f1+f2,
# producing concentric ripples whose amplitude slowly pulses — the
# classic "beat" phenomenon.  The pattern has full SO(2) (circular)
# symmetry.

import numpy as np
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} beating-wave images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    for i in range(n):
        # Two close frequencies
        f_base = random.uniform(0.4, 1.2)
        df = random.uniform(0.03, 0.15)
        f1 = f_base + df / 2
        f2 = f_base - df / 2
        phase = random.uniform(0, 2 * math.pi)

        pattern = np.sin(f1 * r_grid + phase) + np.sin(f2 * r_grid)

        # Soft circular fade
        fade = np.clip(1.0 - r_grid / (size / 2), 0, 1)
        pattern *= fade

        # Normalise to [0, 255]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max - p_min > 0:
            pattern = (pattern - p_min) / (p_max - p_min) * 255
        images[i] = pattern.astype(np.uint8)

    print("Generation complete.")
    return images, labels
