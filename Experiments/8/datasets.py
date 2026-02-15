# Experiment 8: Radial Mandala (n-Fold Rotational + Concentric Rings)
#
# Multiple concentric rings, each decorated with radially-distributed
# dots and connecting spokes.  The fold number (4–12) is randomized per
# image, giving a variety of rotational symmetries.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    """
    Generates N mandala images with n-fold rotational symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} mandala images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        fold = random.choice([4, 5, 6, 7, 8, 10, 12])
        num_rings = random.randint(3, 6)

        for ring in range(num_rings):
            r = 5 + (size / 2 - 7) * (ring + 1) / num_rings

            # Concentric circle
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255)

            # Dots at each fold position
            dot_r = random.uniform(1, 3)
            for k in range(fold):
                angle = 2 * math.pi * k / fold + ring * math.pi / (fold * 2)
                dx = r * math.cos(angle)
                dy = r * math.sin(angle)
                draw.ellipse(
                    [cx + dx - dot_r, cy + dy - dot_r,
                     cx + dx + dot_r, cy + dy + dot_r],
                    fill=255,
                )

                # Connecting spokes to previous ring
                if ring > 0:
                    r_inner = 5 + (size / 2 - 7) * ring / num_rings
                    ix = cx + r_inner * math.cos(angle)
                    iy = cy + r_inner * math.sin(angle)
                    draw.line([(ix, iy), (cx + dx, cy + dy)], fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
