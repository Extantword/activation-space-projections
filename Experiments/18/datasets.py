# Experiment 18: Truchet Tiling (Arc-Based Mosaic)
#
# The image is divided into a grid of square cells.  Each cell contains
# two quarter-circle arcs connecting midpoints of adjacent edges.  The
# tile orientation (one of two types) is chosen randomly on the right
# half and mirrored to the left, enforcing bilateral symmetry.  The
# resulting pattern forms continuous, smoothly curving labyrinths.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def _draw_arc(draw, cx, cy, radius, start_rad, end_rad, steps=15):
    """Draw a circular arc as a polyline."""
    pts = []
    for s in range(steps + 1):
        angle = start_rad + (end_rad - start_rad) * s / steps
        pts.append((cx + radius * math.cos(angle),
                     cy + radius * math.sin(angle)))
    draw.line(pts, fill=255, width=1)


def generate_dataset(n=5000, size=64):
    """
    Generates N Truchet-tiling images with bilateral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} Truchet tiling images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        cs = random.choice([8, 16])  # cell size
        grid = size // cs

        # Choose tile types for right half, mirror to left
        tiles = {}
        for gy in range(grid):
            for gx in range(grid):
                mx = grid - 1 - gx  # mirror column
                if gx >= (grid + 1) // 2:
                    # Use mirrored value (flip type for mirror symmetry)
                    tiles[(gx, gy)] = 1 - tiles.get((mx, gy), 0)
                else:
                    tiles[(gx, gy)] = random.choice([0, 1])

        for (gx, gy), tile_type in tiles.items():
            x0 = gx * cs
            y0 = gy * cs
            r = cs / 2

            if tile_type == 0:
                # Arcs at top-left and bottom-right corners
                _draw_arc(draw, x0, y0, r, 0, math.pi / 2)
                _draw_arc(draw, x0 + cs, y0 + cs, r,
                          math.pi, 3 * math.pi / 2)
            else:
                # Arcs at top-right and bottom-left corners
                _draw_arc(draw, x0 + cs, y0, r,
                          math.pi / 2, math.pi)
                _draw_arc(draw, x0, y0 + cs, r,
                          3 * math.pi / 2, 2 * math.pi)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
