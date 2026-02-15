# Experiment 20: Fermat Spiral / Phyllotaxis (Golden-Angle Symmetry)
#
# Points are placed along a Fermat spiral using the golden angle:
#   r_k = c * sqrt(k),   theta_k = k * golden_angle
# This produces the sunflower-like phyllotactic pattern with approximate
# rotational symmetry related to Fibonacci numbers.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))  # ~137.508 degrees


def generate_dataset(n=5000, size=64):
    """
    Generates N phyllotaxis (Fermat spiral) images.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} phyllotaxis images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        num_points = random.randint(80, 300)
        # Scale factor so outermost point fits in the image
        c = (size / 2 - 4) / math.sqrt(num_points)
        dot_r = random.uniform(0.8, 2.0)
        angle_offset = random.uniform(0, 2 * math.pi)

        for k in range(num_points):
            r = c * math.sqrt(k)
            theta = k * GOLDEN_ANGLE + angle_offset
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                         fill=255)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
