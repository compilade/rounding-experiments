#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

import rounding

# Real behavior of the rounding functions

# [-n .. n]
n = 7

# pixels
M = 512

# Ranges (to allow zooming to a particular region)
# Rx = (0.74, 0.76)
# Ry = (0.54, 0.56)
Rx = (-1, 1)
Ry = (-1, 1)


def round_to_angle(a: np.ndarray) -> np.ndarray:

    min_max = n

    q = rounding.anyrize_inv_sq(a, min_max, axis=-1)
    # q = rounding.absmax_dumb_round(a, min_max, axis=-1)
    # q = rounding.binary_offset(a, axis=-1)
    # q = rounding.anyrize_offset_min_mean(a, min_max, axis=-1)
    # q = rounding.make_qx_quants(min_max, a)

    an = a / np.sqrt(np.sum(np.square(a), axis=-1, keepdims=True))
    qn = q.q / np.sqrt(np.sum(np.square(q.q), axis=-1, keepdims=True))

    qs = np.clip(rounding.np_roundf(a / q.sc), -abs(min_max), abs(min_max))
    qsn = qs / np.sqrt(np.sum(np.square(qs), axis=-1, keepdims=True))

    # return np.sum(an * qn, axis=-1)
    return -np.sum(np.square(a - q.v), axis=-1)
    # Cool geometric shapes
    # return np.sum(qsn * qn, axis=-1)
    # assert q.sc is not None and q.iscale is not None
    # return q.iscale / q.sc

    # return np.sum(
    #     np.square(
    #         rounding.np_roundf(a / q.sc)  #  np.max(abs(a), axis=-1, keepdims=True)),
    #     ),
    #     axis=-1,
    #     keepdims=True,
    # )

    # return q.sc / np.max(abs(a), axis=-1, keepdims=True)
    # np.sqrt(np.sum(a * a, axis=-1, keepdims=True))

    # s = np.sum(a * a, axis=-1, keepdims=True)
    # ss = s - a * a
    # return np.sum(ss, axis=-1, keepdims=True)


plane = np.array(
    [
        [
            Rx[0] + (Rx[1] - Rx[0]) * (i / (M - 1)),
            Ry[0] + (Ry[1] - Ry[0]) * (j / (M - 1)),
            1,
            # 0.73,
            # 0.8,
        ]
        for i in range(M)
        for j in range(M)
    ]
)
cos = round_to_angle(plane)

cos = cos.reshape((M, M))

print(f"{np.min(cos)=}")
print(f"{np.mean(cos)=}")
print(f"{np.max(cos)=}")

plt.figure()
plt.imshow(cos)
plt.show()

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

plt.figure(dpi=96, figsize=(cos.shape[-1] / 96, cos.shape[-2] / 96))
plt.figimage(cos)
plt.savefig(f"images/cube-face-round-{prefix}ary-{2*M}x{2*M}.png")
plt.close()
