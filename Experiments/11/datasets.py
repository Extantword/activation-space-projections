# Experiment 11: Rose Curves (Polar Petal Symmetry)
#
# Rhodonea curves  r = cos(k*theta)  traced in polar coordinates.
# When k is odd the curve has k petals; when k is even it has 2k petals.
# The resulting figure always has C_k or C_{2k} rotational symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N rose-curve images with petal symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} rose curve images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2
        amplitude = size / 2 - 3

        k = random.choice([2, 3, 4, 5, 6, 7, 8])
        phase = random.uniform(0, 2 * math.pi)

        steps = 600
        pts = []
        for s in range(steps + 1):
            theta = 2 * math.pi * s / steps
            r = amplitude * math.cos(k * theta + phase)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            pts.append((x, y))

        draw.line(pts, fill=255, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
