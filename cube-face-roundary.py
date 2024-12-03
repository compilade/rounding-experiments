#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

import rounding

# Real behavior of the rounding functions

# [-n .. n]
n = 3

# pixels
M = 256


def round_to_angle(a: np.ndarray) -> np.ndarray:

    min_max = n

    q = rounding.anyrize_inv_sq(a, min_max, axis=-1)
    # q = rounding.anyrize_offset_min_mean(a, min_max, axis=-1)

    an = a / np.sqrt(np.sum(np.square(a), axis=-1, keepdims=True))
    qn = q.v / np.sqrt(np.sum(np.square(q.v), axis=-1, keepdims=True))

    return np.sum(an * qn, axis=-1)


plane = np.array([[1, i / (M - 1), j / (M - 1)] for i in range(M) for j in range(M)])
cos = round_to_angle(plane)

cos = cos.reshape((M, M))

plt.figure()
plt.imshow(cos)
plt.show()

bottom = np.concatenate([cos[..., ::-1], cos], axis=-1)
face = np.concatenate([bottom[::-1], bottom], axis=-2)

# 2*n + 1
prefix = {
    1: "tern",  # ternary
    2: "pent",  # pentary
    3: "hept",  # heptary
    4: "enne",  # enneary
    5: "hendec",  # hendecary
    7: "pentadec",  # pentadecary
    15: "tricontahen",  # tricontahenary
}.get(n, f"{n}_")

plt.figure(dpi=96, figsize=(face.shape[-1] / 96, face.shape[-2] / 96))
plt.figimage(face)
plt.savefig(f"images/cube-face-round-{prefix}ary-{2*M}x{2*M}.png")
plt.close()
