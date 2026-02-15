# Experiment 33: Rotated Line Grids (Multi-Angle Hatching)
#
# N sets of equally-spaced parallel lines are drawn at uniformly
# separated angles (2*pi/N apart).  The overlay of the grids produces
# interference patterns with C_N rotational symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} rotated line-grid images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        num_sets = random.choice([3, 4, 5, 6])
        spacing = random.uniform(5, 12)
        base_angle = random.uniform(0, 2 * math.pi)

        diag = math.sqrt(2) * size  # max span needed

        for s in range(num_sets):
            angle = base_angle + math.pi * s / num_sets
            dx = math.cos(angle)
            dy = math.sin(angle)
            # Perpendicular direction
            px, py = -dy, dx

            # Number of lines needed to cover the image
            num_lines = int(diag / spacing) + 2
            for j in range(-num_lines, num_lines + 1):
                offset = j * spacing
                # Centre of the line shifted perpendicular by offset
                lx = size / 2 + px * offset
                ly = size / 2 + py * offset
                x1 = lx - dx * diag / 2
                y1 = ly - dy * diag / 2
                x2 = lx + dx * diag / 2
                y2 = ly + dy * diag / 2
                draw.line([(x1, y1), (x2, y2)], fill=180, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
