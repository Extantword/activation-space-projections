# Experiment 5: Lissajous Curves (Parametric Symmetry)
#
# x = sin(a*t + delta), y = sin(b*t) with integer frequency ratios.
# Different (a, b) pairs produce distinct symmetry classes: bilateral,
# rotational, and combined reflective symmetries.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N Lissajous curve images with parametric symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} Lissajous images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    freq_pairs = [
        (1, 2), (1, 3), (2, 3), (3, 4), (3, 5),
        (4, 5), (1, 4), (2, 5), (3, 7), (5, 6),
    ]

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        a, b = random.choice(freq_pairs)
        delta = random.uniform(0, math.pi)
        amplitude = size / 2 - 3
        cx, cy = size / 2, size / 2

        steps = 500
        pts = []
        for s in range(steps + 1):
            t = 2 * math.pi * s / steps
            x = cx + amplitude * math.sin(a * t + delta)
            y = cy + amplitude * math.sin(b * t)
            pts.append((x, y))

        draw.line(pts, fill=255, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
