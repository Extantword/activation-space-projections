# Experiment 15: Epicycloid Curves (Circle Rolling Outside)
#
# Curves traced by a point on a circle of radius r rolling WITHOUT
# slipping on the OUTSIDE of a fixed circle of radius R:
#   x = (R+r)*cos(t) - r*cos((R+r)*t/r)
#   y = (R+r)*sin(t) - r*sin((R+r)*t/r)
# Complementary to the hypotrochoid (Experiment 6) which rolls inside.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N epicycloid images with rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} epicycloid images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        R = random.uniform(8, 20)
        r = random.uniform(3, R * 0.7)
        d = random.uniform(r * 0.3, r * 1.2)

        revolutions = 20
        steps = revolutions * 100

        raw_pts = []
        for s in range(steps + 1):
            t = 2 * math.pi * revolutions * s / steps
            x = (R + r) * math.cos(t) - d * math.cos((R + r) * t / r)
            y = (R + r) * math.sin(t) - d * math.sin((R + r) * t / r)
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
