# Experiment 37: Koch-like Fractal Boundaries
#
# Starting from a regular polygon (3-6 sides), Koch subdivision is
# applied to every edge for 2-4 iterations.  Each iteration replaces
# every straight segment with four segments forming an equilateral
# bump, producing a fractal boundary with D_n symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def _koch_subdivide(points):
    """One iteration of Koch subdivision on a closed polygon."""
    new_pts = []
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        ax = p1[0] + dx / 3
        ay = p1[1] + dy / 3
        bx = p1[0] + 2 * dx / 3
        by = p1[1] + 2 * dy / 3

        # Equilateral triangle peak (outward for CCW polygon)
        px = p1[0] + dx / 2 - dy * math.sqrt(3) / 6
        py = p1[1] + dy / 2 + dx * math.sqrt(3) / 6

        new_pts.extend([(p1[0], p1[1]), (ax, ay), (px, py), (bx, by)])
    return new_pts


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} Koch fractal images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        sides = random.choice([3, 4, 5, 6])
        iterations = random.randint(2, 4)
        phase = random.uniform(0, 2 * math.pi)

        # Initial regular polygon (CCW winding)
        verts = []
        for v in range(sides):
            angle = 2 * math.pi * v / sides + phase
            verts.append((math.cos(angle), math.sin(angle)))

        # Apply Koch iterations
        pts = list(verts)
        for _ in range(iterations):
            pts = _koch_subdivide(pts)

        # Scale to fit
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        rng = max(max_x - min_x, max_y - min_y)
        if rng == 0:
            rng = 1
        sc = (size - 6) / rng
        cx, cy = size / 2, size / 2
        mx, my = (min_x + max_x) / 2, (min_y + max_y) / 2

        scaled = [(cx + (x - mx) * sc, cy + (y - my) * sc) for x, y in pts]
        scaled.append(scaled[0])  # close the curve
        draw.line(scaled, fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
