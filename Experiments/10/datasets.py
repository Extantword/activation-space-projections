# Experiment 10: Star Polygons (Rotational Symmetry)
#
# Star polygons {n/k} are formed by connecting every k-th vertex of a
# regular n-gon.  Multiple nested star polygons at different scales
# and step sizes produce rich rotationally-symmetric figures.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N star-polygon images with rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} star polygon images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        n_vertices = random.randint(5, 15)
        num_stars = random.randint(1, 3)
        base_rotation = random.uniform(0, 2 * math.pi)

        for s in range(num_stars):
            r = (size / 2 - 2) * (1 - s * 0.25)
            if r < 5:
                break
            k = random.randint(2, max(2, n_vertices // 2 - 1))
            rotation = base_rotation + s * random.uniform(0, math.pi / n_vertices)

            # Vertices of the regular n-gon
            vertices = []
            for v in range(n_vertices):
                angle = rotation + 2 * math.pi * v / n_vertices
                vertices.append((cx + r * math.cos(angle),
                                 cy + r * math.sin(angle)))

            # Connect every k-th vertex
            for v in range(n_vertices):
                v_next = (v + k) % n_vertices
                draw.line([vertices[v], vertices[v_next]], fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
