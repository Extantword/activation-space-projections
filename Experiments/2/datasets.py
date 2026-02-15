# Experiment 2: Bilateral Butterfly (Vertical Mirror Symmetry)
#
# Random cubic Bezier curves are drawn on the right half of the image,
# then mirrored across the vertical center axis, producing butterfly-like
# shapes with exact bilateral symmetry.

import numpy as np
from PIL import Image, ImageDraw
import random


def _cubic_bezier(t, p0, p1, p2, p3):
    """Evaluate cubic Bezier at parameter t."""
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def generate_dataset(n=5000, size=64):
    """
    Generates N butterfly images with bilateral (vertical mirror) symmetry.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} butterfly images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    cx = size // 2

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        num_curves = random.randint(2, 5)
        for _ in range(num_curves):
            # Endpoints anchored on the center axis
            p0 = (cx, random.randint(5, size - 5))
            p3 = (cx, random.randint(5, size - 5))
            # Control points on the right half
            p1 = (random.randint(cx, size - 1), random.randint(0, size - 1))
            p2 = (random.randint(cx, size - 1), random.randint(0, size - 1))

            steps = 50
            right_pts = [_cubic_bezier(t / steps, p0, p1, p2, p3)
                         for t in range(steps + 1)]
            left_pts = [(2 * cx - x, y) for x, y in right_pts]

            draw.line(right_pts, fill=255, width=1)
            draw.line(left_pts, fill=255, width=1)

        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
