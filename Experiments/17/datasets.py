# Experiment 17: Symmetric Dot Clouds (D_n Dihedral Point Patterns)
#
# Random dots are placed inside a fundamental wedge and then replicated
# by the full dihedral group D_n (rotations + reflections), producing
# a symmetric scatter of filled circles.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N symmetric dot-cloud images with D_n symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} symmetric dot-cloud images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        fold = random.choice([3, 4, 5, 6, 8])
        num_dots = random.randint(8, 30)
        dot_r = random.uniform(1.0, 2.5)

        # Generate dots in the fundamental wedge [0, pi/fold]
        base_dots = []
        for _ in range(num_dots):
            r = random.uniform(3, size / 2 - 3)
            a = random.uniform(0, math.pi / fold)
            base_dots.append((r, a))

        # Replicate with D_n: n rotations x 2 mirrors
        for r, a in base_dots:
            for k in range(fold):
                offset = k * 2 * math.pi / fold
                for mirror in [1, -1]:
                    angle = mirror * a + offset
                    dx = cx + r * math.cos(angle)
                    dy = cy + r * math.sin(angle)
                    draw.ellipse(
                        [dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r],
                        fill=255,
                    )

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
