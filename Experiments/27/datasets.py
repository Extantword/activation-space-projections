# Experiment 27: Tangent-Line Envelopes (Curve Envelopes)
#
# A family of straight lines tangent to a hidden circle is drawn.
# Each line is parameterised by an angle; the envelope of the family
# (the curve all lines are tangent to) is the circle itself, but the
# collection of tangent lines fills the image with a characteristic
# symmetric pattern.  Multiple concentric tangent families create
# richer figures.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N tangent-line envelope images with rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} tangent envelope images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        circle_r = random.uniform(8, size * 0.3)
        num_lines = random.randint(20, 60)
        line_half_len = size * 0.8

        for j in range(num_lines):
            theta = 2 * math.pi * j / num_lines
            # Point of tangency on the circle
            tx = cx + circle_r * math.cos(theta)
            ty = cy + circle_r * math.sin(theta)
            # Tangent direction (perpendicular to radius)
            dx = -math.sin(theta)
            dy = math.cos(theta)
            # Line endpoints
            x1 = tx + line_half_len * dx
            y1 = ty + line_half_len * dy
            x2 = tx - line_half_len * dx
            y2 = ty - line_half_len * dy
            draw.line([(x1, y1), (x2, y2)], fill=180, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
