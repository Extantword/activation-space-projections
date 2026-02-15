# Experiment 32: Concentric Dot Rings (Radial Lattice)
#
# Dots are placed on concentric rings of increasing radius.  Each ring
# has fold * ring_index equally-spaced dots, so inner rings are sparser
# and outer rings denser.  The global pattern has D_{fold} symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} concentric dot-ring images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        fold = random.choice([3, 4, 5, 6, 8])
        num_rings = random.randint(3, 7)
        max_r = size / 2 - 3
        dot_r = random.uniform(0.8, 2.0)
        phase = random.uniform(0, 2 * math.pi)

        for ring in range(1, num_rings + 1):
            r = max_r * ring / num_rings
            num_dots = fold * ring
            for d in range(num_dots):
                angle = 2 * math.pi * d / num_dots + phase
                dx = cx + r * math.cos(angle)
                dy = cy + r * math.sin(angle)
                draw.ellipse([dx - dot_r, dy - dot_r,
                              dx + dot_r, dy + dot_r], fill=255)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
