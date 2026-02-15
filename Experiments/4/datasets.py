# Experiment 4: Concentric Polygons (Rotational Symmetry)
#
# Nested regular polygons of the same type (e.g. all pentagons) drawn
# at decreasing radii with optional incremental rotation, producing
# Cn rotational symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N concentric polygon images with rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} concentric polygon images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        sides = random.randint(3, 8)
        num_rings = random.randint(3, 7)
        base_rotation = random.uniform(0, 2 * math.pi)
        rotation_increment = random.uniform(
            -math.pi / (2 * sides), math.pi / (2 * sides)
        )

        for ring in range(num_rings):
            r = (size / 2 - 2) * (1 - ring / num_rings)
            if r < 3:
                break
            rotation = base_rotation + ring * rotation_increment
            pts = []
            for v in range(sides):
                angle = rotation + 2 * math.pi * v / sides
                pts.append((cx + r * math.cos(angle),
                            cy + r * math.sin(angle)))
            pts.append(pts[0])  # close polygon
            draw.line(pts, fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
