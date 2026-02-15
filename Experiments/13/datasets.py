# Experiment 13: Symmetric Binary Trees (Bilateral Branching)
#
# A recursive binary tree where left and right branches always use the
# same angle and shrinkage factor, producing exact bilateral symmetry.
# The tree grows upward from the bottom centre of the image.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def _draw_tree(draw, x, y, angle, length, depth, max_depth,
               branch_angle, shrink):
    """Recursively draw a symmetric binary tree."""
    if depth > max_depth or length < 1.5:
        return
    ex = x + length * math.cos(angle)
    ey = y + length * math.sin(angle)
    draw.line([(x, y), (ex, ey)], fill=255, width=1)
    _draw_tree(draw, ex, ey, angle - branch_angle,
               length * shrink, depth + 1, max_depth, branch_angle, shrink)
    _draw_tree(draw, ex, ey, angle + branch_angle,
               length * shrink, depth + 1, max_depth, branch_angle, shrink)


def generate_dataset(n=5000, size=64):
    """
    Generates N symmetric binary-tree images with bilateral symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} symmetric tree images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        trunk_length = random.uniform(size * 0.12, size * 0.22)
        max_depth = random.randint(5, 8)
        branch_angle = random.uniform(math.pi / 8, math.pi / 3)
        shrink = random.uniform(0.6, 0.78)

        start_x = size / 2
        start_y = size - 3
        _draw_tree(draw, start_x, start_y, -math.pi / 2,
                   trunk_length, 0, max_depth, branch_angle, shrink)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
