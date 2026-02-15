# Experiment 25: Limacon Curves (Polar Curve Family)
#
# The limacon is a family of polar curves:  r = a + b*cos(theta).
# Depending on the ratio a/b the shape ranges from a cardioid (a=b)
# to a limacon with an inner loop (a < b), a dimpled limacon, or a
# convex curve.  All limacons have bilateral (mirror) symmetry about
# the polar axis.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N limacon images with bilateral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} limacon images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        a = random.uniform(0.3, 1.5)
        b = random.uniform(0.5, 1.5)
        rotation = random.uniform(0, 2 * math.pi)

        steps = 500
        raw_pts = []
        for s in range(steps + 1):
            theta = 2 * math.pi * s / steps
            r = a + b * math.cos(theta)
            x = r * math.cos(theta + rotation)
            y = r * math.sin(theta + rotation)
            raw_pts.append((x, y))

        # Scale to fit
        xs = [p[0] for p in raw_pts]
        ys = [p[1] for p in raw_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        scale = (size - 6) / max(range_x, range_y)

        pts = []
        for x, y in raw_pts:
            px = cx + (x - (min_x + max_x) / 2) * scale
            py = cy + (y - (min_y + max_y) / 2) * scale
            pts.append((px, py))

        draw.line(pts, fill=255, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
