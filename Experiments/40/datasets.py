# Experiment 40: Spiraling Regular Polygons (Polygon Vortex)
#
# Nested regular k-gons are drawn at exponentially-decreasing radii
# with a constant angular step between successive layers, creating a
# spiral-vortex effect.  Unlike Experiment 4 (linear radius, small
# rotation), this uses exponential shrinkage and larger twist, so the
# polygons converge dramatically toward the centre.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} spiraling-polygon images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        sides = random.choice([3, 4, 5, 6, 8])
        num_layers = random.randint(10, 30)
        shrink = random.uniform(0.85, 0.95)
        twist_per_layer = random.uniform(math.pi / 20, math.pi / 5)
        base_rotation = random.uniform(0, 2 * math.pi)

        r = size / 2 - 2
        for layer in range(num_layers):
            rotation = base_rotation + layer * twist_per_layer
            pts = []
            for v in range(sides):
                angle = rotation + 2 * math.pi * v / sides
                pts.append((cx + r * math.cos(angle),
                            cy + r * math.sin(angle)))
            pts.append(pts[0])  # close
            draw.line(pts, fill=255, width=1)
            r *= shrink
            if r < 2:
                break

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
