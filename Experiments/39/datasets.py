# Experiment 39: Harmonic Orbits (Multi-Frequency Circular Motion)
#
# The trace of a sum of K circular motions at integer frequencies:
#   z(t) = sum_k  A_k * exp(i * f_k * t)
# This generalises the 2-frequency spirograph to 3-5 harmonics,
# producing intricate closed curves whose symmetry depends on the
# GCD of the frequencies.

import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_dataset(n=5000, size=64):
    print(f"Generating {n} harmonic-orbit images ({size}x{size})...")
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2

        num_harmonics = random.randint(3, 5)

        # Random integer frequencies and amplitudes
        freqs = sorted(random.sample(range(1, 15), num_harmonics))
        amps = [random.uniform(0.3, 1.0) for _ in range(num_harmonics)]
        phases = [random.uniform(0, 2 * math.pi) for _ in range(num_harmonics)]

        # Trace the orbit
        steps = 2000
        raw = []
        for s in range(steps + 1):
            t = 2 * math.pi * s / steps
            x = sum(a * math.cos(f * t + p)
                    for a, f, p in zip(amps, freqs, phases))
            y = sum(a * math.sin(f * t + p)
                    for a, f, p in zip(amps, freqs, phases))
            raw.append((x, y))

        # Scale to fit
        xs = [p[0] for p in raw]
        ys = [p[1] for p in raw]
        rng = max(max(xs) - min(xs), max(ys) - min(ys))
        if rng == 0:
            rng = 1
        sc = (size - 6) / rng
        mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        pts = [(cx + (x - mx) * sc, cy + (y - my) * sc) for x, y in raw]

        draw.line(pts, fill=255, width=1)
        images[i] = np.array(img)

    print("Generation complete.")
    return images, labels
