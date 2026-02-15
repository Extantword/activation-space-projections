# Experiment 35: Overlapping Circles (Venn-Ring Patterns)
#
# N circles of equal radius are centred on the vertices of a regular
# N-gon.  Only the circle outlines are drawn, so the overlapping
# regions create petal- and lens-shaped intersection arcs.  The
# arrangement has D_N dihedral symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} overlapping-circle images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        num_circles = random.choice([3, 4, 5, 6, 7, 8])
        ring_r = random.uniform(size * 0.1, size * 0.25)
        circle_r = random.uniform(size * 0.15, size * 0.35)
        phase = random.uniform(0, 2 * math.pi)

        for k in range(num_circles):
            angle = 2 * math.pi * k / num_circles + phase
            ox = cx + ring_r * math.cos(angle)
            oy = cy + ring_r * math.sin(angle)
            bbox = [ox - circle_r, oy - circle_r,
                    ox + circle_r, oy + circle_r]
            draw.ellipse(bbox, outline=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
