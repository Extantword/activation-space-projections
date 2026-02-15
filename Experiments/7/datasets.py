# Experiment 7: Symmetric Random Walk (Bilateral Mirror Symmetry)
#
# A random walk wanders on the right half of the image, then the entire
# path is mirrored across the vertical center axis.  The result resembles
# Rorschach inkblots — organic yet perfectly bilaterally symmetric.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N symmetric random-walk images with bilateral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} symmetric random-walk images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    cx = size // 2

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        # Start from center
        x, y = float(cx), float(size // 2)
        step_size = random.uniform(1.5, 3.0)
        num_steps = random.randint(80, 200)

        right_pts = [(x, y)]
        for _ in range(num_steps):
            angle = random.uniform(-math.pi / 2, math.pi / 2)  # bias rightward
            dx = step_size * math.cos(angle)
            dy = step_size * math.sin(angle)
            x = max(cx, min(size - 1, x + dx))
            y = max(0, min(size - 1, y + dy))
            right_pts.append((x, y))

        # Mirror across vertical axis
        left_pts = [(2 * cx - px, py) for px, py in right_pts]

        draw.line(right_pts, fill=255, width=1)
        draw.line(left_pts, fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
