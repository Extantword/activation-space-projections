# Experiment 23: Nested Archimedean Spirals (C_n Rotational Symmetry)
#
# N Archimedean spirals (r = a + b*theta) are drawn with equal angular
# offsets of 2*pi/N, producing a pattern with C_N rotational symmetry.
# The spirals wind outward from the centre, creating a pinwheel effect.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N nested-spiral images with C_N rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} nested spiral images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        num_arms = random.choice([2, 3, 4, 5, 6])
        max_r = size / 2 - 2
        turns = random.uniform(1.5, 4.0)
        max_theta = turns * 2 * math.pi
        b = max_r / max_theta  # spacing between turns

        steps = 300
        for arm in range(num_arms):
            offset = arm * 2 * math.pi / num_arms
            pts = []
            for s in range(steps + 1):
                theta = max_theta * s / steps
                r = b * theta
                x = cx + r * math.cos(theta + offset)
                y = cy + r * math.sin(theta + offset)
                pts.append((x, y))
            draw.line(pts, fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
