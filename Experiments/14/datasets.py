# Experiment 14: Guilloche Patterns (Overlapping Spirographs)
#
# Two hypotrochoid curves drawn with slightly different pen distances
# and a phase offset.  The overlay of the two creates the intricate
# interlocking patterns seen on banknotes and security documents.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def _hypotrochoid(R, r, d, phase, revolutions, steps):
    """Return a list of (x, y) points for a hypotrochoid."""
    pts = []
    for s in range(steps + 1):
        t = 2 * math.pi * revolutions * s / steps + phase
        x = (R - r) * math.cos(t) + d * math.cos((R - r) * t / r)
        y = (R - r) * math.sin(t) - d * math.sin((R - r) * t / r)
        pts.append((x, y))
    return pts


def _scale_and_centre(raw_pts, size, margin=3):
    """Scale a point list to fit within (size x size) with margin."""
    xs = [p[0] for p in raw_pts]
    ys = [p[1] for p in raw_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1
    scale = (size - 2 * margin) / max(range_x, range_y)
    cx, cy = size / 2, size / 2
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [(cx + (x - mid_x) * scale, cy + (y - mid_y) * scale)
            for x, y in raw_pts]


def generate_dataset(n=5000, size=64):
    """
    Generates N guilloche images from overlapping hypotrochoids.
    Returns:
        images: numpy array of shape (N, size, size), dtype uint8
        labels: numpy array of shape (N,), dtype int64
    """
    print(f"Generating {n} guilloche images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)

        R = random.uniform(14, 25)
        r = random.uniform(3, R * 0.6)
        d1 = random.uniform(r * 0.4, r * 1.0)
        d2 = random.uniform(r * 0.4, r * 1.0)
        phase_offset = random.uniform(math.pi / 12, math.pi / 3)

        revolutions = 20
        steps = 2000

        # Compute both curves in the same coordinate system
        raw1 = _hypotrochoid(R, r, d1, 0, revolutions, steps)
        raw2 = _hypotrochoid(R, r, d2, phase_offset, revolutions, steps)

        # Find global bounding box for consistent scaling
        all_pts = raw1 + raw2
        pts1 = _scale_and_centre(raw1, size)
        # Re-scale raw2 using same box — approximate via combined scaling
        pts2 = _scale_and_centre(raw2, size)

        draw.line(pts1, fill=200, width=1)
        draw.line(pts2, fill=200, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
