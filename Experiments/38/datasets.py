# Experiment 38: Superformula Curves (Gielis Shapes)
#
# The Gielis superformula defines a polar curve:
#   r(theta) = [ |cos(m*theta/4)/a|^n2 + |sin(m*theta/4)/b|^n3 ]^(-1/n1)
# Varying the parameters produces circles, polygons, stars, flowers,
# and many organic-looking shapes, all with C_m rotational symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def _superformula_r(theta, m, a, b, n1, n2, n3):
    """Evaluate the Gielis superformula at angle theta."""
    t = m * theta / 4.0
    term1 = abs(math.cos(t) / a) ** n2
    term2 = abs(math.sin(t) / b) ** n3
    s = term1 + term2
    if s == 0:
        return 0.0
    return s ** (-1.0 / n1)


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} superformula images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        m = random.choice([3, 4, 5, 6, 7, 8])
        a = random.uniform(0.8, 1.2)
        b = random.uniform(0.8, 1.2)
        n1 = random.uniform(0.3, 6.0)
        n2 = random.uniform(0.5, 6.0)
        n3 = random.uniform(0.5, 6.0)
        rotation = random.uniform(0, 2 * math.pi)

        steps = 800
        raw = []
        for s in range(steps + 1):
            theta = 2 * math.pi * s / steps
            r = _superformula_r(theta, m, a, b, n1, n2, n3)
            x = r * math.cos(theta + rotation)
            y = r * math.sin(theta + rotation)
            raw.append((x, y))

        # Scale to fit
        xs = [p[0] for p in raw]
        ys = [p[1] for p in raw]
        rng = max(max(xs) - min(xs), max(ys) - min(ys))
        if rng == 0:
            rng = 1
        sc = (size - 6) / rng
        mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        pts = [(cx + (x - mx) * sc, cy + (y - my) * sc) for x, y in raw]

        draw.line(pts, fill=255, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
