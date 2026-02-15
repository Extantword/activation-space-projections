# Experiment 3: Snowflake Dendrites (6-Fold Rotational + Fractal Branching)
#
# A single recursive branching structure is generated with a fixed random
# seed, then drawn 6 times at 60-degree intervals to create a snowflake
# with C6 rotational symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def _draw_branch(draw, cx, cy, angle, length, depth, max_depth):
    """Recursively draw a branching dendrite."""
    if depth > max_depth or length < 2:
        return
    ex = cx + length * math.cos(angle)
    ey = cy + length * math.sin(angle)
    draw.line([(cx, cy), (ex, ey)], fill=255, width=1)

    # Sub-branches at random angles
    branch_angle = random.uniform(math.pi / 6, math.pi / 3)
    shrink = random.uniform(0.45, 0.65)
    branch_pos = random.uniform(0.4, 0.8)

    bx = cx + length * branch_pos * math.cos(angle)
    by = cy + length * branch_pos * math.sin(angle)

    _draw_branch(draw, bx, by, angle + branch_angle, length * shrink, depth + 1, max_depth)
    _draw_branch(draw, bx, by, angle - branch_angle, length * shrink, depth + 1, max_depth)

    # Tip sub-branches
    tip_angle = random.uniform(math.pi / 8, math.pi / 4)
    tip_shrink = random.uniform(0.3, 0.5)
    _draw_branch(draw, ex, ey, angle + tip_angle, length * tip_shrink, depth + 1, max_depth)
    _draw_branch(draw, ex, ey, angle - tip_angle, length * tip_shrink, depth + 1, max_depth)


def generate_dataset(n=5000, size=64):
    """
    Generates N snowflake images with 6-fold rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} snowflake images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        branch_length = random.uniform(size * 0.3, size * 0.45)
        max_depth = random.randint(2, 3)

        # Fix seed so all 6 arms are identical
        branch_seed = random.randint(0, 10**8)

        for k in range(6):
            angle = k * math.pi / 3
            random.seed(branch_seed)
            _draw_branch(draw, cx, cy, angle, branch_length, 0, max_depth)

        # Restore true randomness for next image
        random.seed()

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
