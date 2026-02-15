# Experiment 1: Kaleidoscope Patterns (D6 Dihedral Symmetry)
#
# Random line segments drawn in one 60-degree wedge, then reflected
# and rotated 6 times to produce full hexagonal (D6) symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N kaleidoscope images with 6-fold dihedral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} kaleidoscope images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2
        max_r = size / 2 - 1

        # Random strokes confined to the base wedge [0, pi/6]
        num_strokes = random.randint(3, 7)
        strokes = []
        for _ in range(num_strokes):
            r1 = random.uniform(3, max_r)
            r2 = random.uniform(3, max_r)
            a1 = random.uniform(0, math.pi / 6)
            a2 = random.uniform(0, math.pi / 6)
            strokes.append((r1, a1, r2, a2))

        # Replicate across 6 rotations x 2 mirrors = 12 copies
        for k in range(6):
            offset = k * math.pi / 3
            for r1, a1, r2, a2 in strokes:
                for mirror in [1, -1]:
                    x1 = cx + r1 * math.cos(mirror * a1 + offset)
                    y1 = cy + r1 * math.sin(mirror * a1 + offset)
                    x2 = cx + r2 * math.cos(mirror * a2 + offset)
                    y2 = cy + r2 * math.sin(mirror * a2 + offset)
                    draw.line([(x1, y1), (x2, y2)], fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
