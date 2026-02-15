# Experiment 28: Frieze Patterns (1-D Translational Symmetry)
#
# A small random motif is drawn inside a single cell and then tiled
# horizontally across the image.  Each cell is also given internal
# bilateral symmetry (left-right mirror within the cell), producing
# one of the classic frieze-group symmetry types.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N frieze-pattern images with translational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} frieze pattern images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        cell_w = random.choice([8, 16, 32])
        num_cells = size // cell_w

        # Create a random motif in the right half of one cell
        num_strokes = random.randint(2, 5)
        motif_strokes = []
        for _ in range(num_strokes):
            x1 = random.uniform(cell_w / 2, cell_w - 1)
            y1 = random.uniform(2, size - 2)
            x2 = random.uniform(cell_w / 2, cell_w - 1)
            y2 = random.uniform(2, size - 2)
            motif_strokes.append((x1, y1, x2, y2))

        # Tile across the image
        for cell in range(num_cells):
            x_off = cell * cell_w
            for x1, y1, x2, y2 in motif_strokes:
                # Right half of cell
                draw.line([(x_off + x1, y1), (x_off + x2, y2)],
                          fill=255, width=1)
                # Mirrored left half
                mx1 = cell_w - x1
                mx2 = cell_w - x2
                draw.line([(x_off + mx1, y1), (x_off + mx2, y2)],
                          fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
