# Experiment 31: Hypocycloid (Cusped Curves)
#
# A point on the rim of a circle of radius r rolling INSIDE a fixed
# circle of radius R traces a hypocycloid.  When R/r = k (integer)
# the curve has k cusps and D_k symmetry.
#   k=3 : deltoid      k=4 : astroid      k=5+ : higher cusps

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} hypocycloid images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        k = random.choice([3, 4, 5, 6, 7, 8])
        R = 1.0
        r = R / k
        rotation = random.uniform(0, 2 * math.pi)

        steps = 800
        raw = []
        for s in range(steps + 1):
            t = 2 * math.pi * s / steps
            x = (R - r) * math.cos(t) + r * math.cos((R - r) * t / r)
            y = (R - r) * math.sin(t) - r * math.sin((R - r) * t / r)
            # Apply global rotation
            xr = x * math.cos(rotation) - y * math.sin(rotation)
            yr = x * math.sin(rotation) + y * math.cos(rotation)
            raw.append((xr, yr))

        # Scale to fit
        xs = [p[0] for p in raw]
        ys = [p[1] for p in raw]
        rng = max(max(xs) - min(xs), max(ys) - min(ys))
        if rng == 0:
            rng = 1
        sc = (size - 6) / rng
        pts = [(cx + (x - (min(xs) + max(xs)) / 2) * sc,
                cy + (y - (min(ys) + max(ys)) / 2) * sc) for x, y in raw]

        draw.line(pts, fill=255, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
