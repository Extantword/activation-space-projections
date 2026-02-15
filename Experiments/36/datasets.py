# Experiment 36: Complete Graph on Regular n-gon (K_n Star Wiring)
#
# n vertices are placed on a regular polygon and every pair is connected
# by a straight line, drawing the complete graph K_n.  The result is a
# dense web of chords with D_n dihedral symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} complete-graph images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        num_v = random.randint(5, 18)
        radius = size / 2 - 3
        phase = random.uniform(0, 2 * math.pi)

        verts = []
        for v in range(num_v):
            angle = 2 * math.pi * v / num_v + phase
            verts.append((cx + radius * math.cos(angle),
                          cy + radius * math.sin(angle)))

        # Draw all C(num_v, 2) edges
        brightness = max(40, min(255, int(600 / num_v)))
        for a in range(num_v):
            for b in range(a + 1, num_v):
                draw.line([verts[a], verts[b]], fill=brightness, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
