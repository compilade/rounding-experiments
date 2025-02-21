#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

import rounding
import rounding_c

# Real behavior of the rounding functions

# [-n .. n]
n = 15

# pixels
M = 1024
M = 1025
M = 2048

# Ranges (to allow zooming to a particular region)
# Rx = (0.74, 0.76)
# Ry = (0.54, 0.56)
Rx = (-0.05, 0.05)
Ry = (-0.8, -0.7)
# Rx = (-1, 1)
# Ry = (-1, 1)
Rx = (0, 1)
Ry = (0, 1)
# Rx = (-0.125, 0.125)
# Ry = (-0.125, 0.125)
# Rx = (0, 1/4)
# Ry = (0, 1/16)
# Ry = (5/511, 6/511)
# Rx = (2/511, 3/511)
# Rx = (0.04, 0.05)
# Ry = (0.22, 0.23)


# problem = [0.05, 0.225, 1]


def normalize(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return v / np.sum(w * v * v, axis=-1, keepdims=True)


def center(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    # return v - np.mean(v, axis=-1, keepdims=True)
    # return v - np.min(v, axis=-1, keepdims=True)
    return v - np.sum(w * v, axis=-1, keepdims=True) / np.sum(w, axis=-1, keepdims=True)


def round_to_angle(a: np.ndarray) -> np.ndarray:

    min_max = n

    # qw = np.sqrt(
    #     np.sum(np.square(a), axis=-1, keepdims=True)
    #     / np.sum(np.ones_like(a), axis=-1, keepdims=True)
    # ) + np.abs(a)
    qw = np.square(a)
    # qw = np.ones_like(a)
    # qw = None

    v = rounding_c.anyrize_qx(a, min_max, qw)
    # v = rounding_c.anyrize_qkxs(a, -min_max, min_max - 1, qw, signed_scale=True)
    # v = rounding_c.anyrize_iq4nl(a, qw=qw)
    # v = rounding_c.anyrize_q3(a, min_max)
    # v = rounding_c.anyrize_qp(a, min_max, qw=qw)
    # v = rounding_c.anyrize_qkxs_iq4nl(a, qw=qw)
    # v = rounding_c.anyrize_qkxs_iq4nl_signed(a, qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-53, -43, -37, -29, -19, -13, -7, -3, 1, 5, 11, 17, 23, 31, 41, 47]), qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-63, -40, -23, -10, 1, 13, 28,  47]), qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-59, -36, -19,  -6, 5, 17, 32,  51]), qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-19, -13, -7, -3, 1, 5, 11, 17]), qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-7, -3, 1, 5]), qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-31, -13, 1, 17]), qw=qw)
    # v = rounding_c.anyrize_qkxs_iqxnl_signed(a, np.array([-26, -8, 6, 22]), qw=qw)
    # v = rounding_c.anyrize_qkx2_q4_k(a, min_max)
    # v = rounding_c.anyrize_qkx3_q4_k(a, min_max)
    # v = rounding_c.anyrize_qkxcm_q4_k(a, min_max)

    # q = rounding.anyrize_inv_sq(a, min_max, axis=-1)
    # q = rounding.absmax_dumb_round(a, min_max, axis=-1)
    # q = rounding.binary_offset(a, axis=-1)
    # q = rounding.binary_offset_mean(a, qw, axis=-1)
    # q = rounding.anyrize_offset_min_mean(a, min_max, axis=-1, w=qw)
    # q = rounding.offset_dumb_round(a, min_max, axis=-1)
    # print(f"{a.shape=}")
    # print(f"{q.q.shape=}")
    # print(f"{q.v.shape=}")
    # q = rounding.make_qx_quants(min_max, a)
    # v = q.v

    # a = a * np.square(a)
    # q.v = q.v * np.square(a)
    qw = np.square(a) if qw is None else qw
    sumlx = np.sum(qw * a * v, axis=-1, keepdims=True)
    suml2 = np.sum(qw * v * v, axis=-1, keepdims=True)
    sumx2 = np.sum(qw * a * a, axis=-1, keepdims=True)
    # return sumlx / suml2
    return np.clip(sumlx / np.sqrt(suml2 * sumx2), -1.0, 1.0)

    # --- abs offset comparison ---
    v = a.copy()
    v[:] = np.array([0.1, -0.1, 1])

    qw = np.square(a) if qw is None else qw
    sumw = np.sum(qw, axis=-1, keepdims=True)
    sumx = np.sum(qw * a, axis=-1, keepdims=True)
    suml = np.sum(qw * v, axis=-1, keepdims=True)
    vm = v - suml / sumw
    xm = a - sumx / sumw
    sumlx = np.sum(qw * a * v, axis=-1, keepdims=True)
    sumlxm = np.sum(qw * xm * vm, axis=-1, keepdims=True)
    suml2 = np.sum(qw * v * v, axis=-1, keepdims=True)
    suml2m = np.sum(qw * vm * vm, axis=-1, keepdims=True)
    sumx2 = np.sum(qw * a * a, axis=-1, keepdims=True)
    sumx2m = np.sum(qw * xm * xm, axis=-1, keepdims=True)
    coss = (sumlx) / np.sqrt(suml2 * sumx2)
    cosm = (sumlxm) / np.sqrt(suml2m * sumx2m)
    D = sumw * suml2 - suml * suml
    this_min = (suml2 * sumx - suml * sumlx) / D
    this_scale = np.where(
        this_min < 0.0, (sumw * sumlx - sumx * suml) / D, sumlx / suml2
    )
    other = v * this_scale + this_min
    # min = mean_x - scale * mean_l
    # other / this_scale = v + this_min / this_scale
    # other / this_scale = L + (suml2 * sumx - suml * sumlx) / (sumw * sumlx - sumx * suml)
    # other * suml2m / sumlxm = L + (mean_x - mean_l * sumlxm / suml2m) * suml2m / sumlxm
    # other / this_scale = L - mean_l + mean_x / this_scale
    # other * suml2m / sumlxm = L - suml/sumw + (sumx/sumw) / (sumlxm / suml2m)
    # other * suml2m / sumlxm = L - suml/sumw + (sumx * suml2m) / (sumlxm * sumw)
    # other / this_scale - mean_x / this_scale = L - mean_l
    # other - mean_x = (L - mean_l) * this_scale
    # other = (L - mean_l) * this_scale + mean_x
    # other
    # normalize(other - mean(other)) = normalize(v - mean(v))
    # return np.sum(np.square(normalize(center(other, qw), qw) - normalize(center(v, qw), qw)), axis=-1, keepdims=True)
    # return this_scale - sumlxm/suml2m
    # How to simplify this distance metric?
    # Need to assume the best scale and offset are used,
    # which means normalizing and centering both? In what order (is this important?)
    # Normalizing *after* centering should be correct?
    # return np.clip(np.sum(qw * a * other, axis=-1, keepdims=True) / np.sqrt(np.sum(qw * a * a, axis=-1, keepdims=True) * np.sum(qw * other * other, axis=-1, keepdims=True)), 0.9, 1.0)
    # The sum of squared error is equivalent to 2 - 2*(cosine similarity) for normalized vectors
    # Cos(A, B) = (sum(A*A) + sum(B*B) - sum((A - B)**2))/(2*sqrt(sum(A*A)*sum(B*B)))
    # 2*sum(A*B) = (sum(A*A) + sum(B*B) - sum((A - B)**2))
    # sum((A - B)**2) = sum(A*A) - 2*sum(A*B) + sum(B*B)
    return np.clip(
        -np.sum(
            qw
            * np.square(
                a / np.sqrt(np.sum(a * a, axis=-1, keepdims=True))
                - other / np.sqrt(np.sum(other * other, axis=-1, keepdims=True))
            ),
            axis=-1,
            keepdims=True,
        )
        / 2
        + 1,
        0.9,
        1.0,
    )
    # float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
    # float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
    # where can the min be used?
    # anywhere it's going to be negative
    # when does that happen?
    # When the quantized vector is closer to 1 than the other?
    # return np.clip(np.where(suml * sumlx > sumx * suml2, np.where(cosm > coss, cosm, coss), coss), 0.9, 1.0)
    # Or is there some other thing I'm not considering?
    # Like how it fades? Oh, right. That's definitely different. Why though?
    # return np.where(cosm > coss, cosm, coss)
    # ---

    # an = a / np.sqrt(np.sum(np.square(a), axis=-1, keepdims=True))
    # qn = q.v / np.sqrt(np.sum(np.square(q.v), axis=-1, keepdims=True))

    # qs = np.clip(rounding.np_roundf(a / q.sc), -abs(min_max), abs(min_max))
    # qsn = qs / np.sqrt(np.sum(np.square(qs), axis=-1, keepdims=True))

    # return np.sum(qn * an, axis=-1)
    # return -np.sum(np.square(np.square(a) - np.square(v)), axis=-1)
    # return np.clip(-np.sum(np.square(a - v), axis=-1), -0.25, 0)
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
    return np.max(abs(a), axis=-1, keepdims=True) / q.sc
    # np.sqrt(np.sum(a * a, axis=-1, keepdims=True))

    # s = np.sum(a * a, axis=-1, keepdims=True)
    # ss = s - a * a
    # return np.sum(ss, axis=-1, keepdims=True)


rng = np.random.default_rng(42)
a = rng.normal(size=(30,)).tolist()


plane = np.array(
    [
        [
            Rx[0] + (Rx[1] - Rx[0]) * (i / (M - 1)),
            Ry[0] + (Ry[1] - Ry[0]) * (j / (M - 1)),
            1,
            # # 0.73,
            # # 0.8,
        ]
        # + [1] * 29
        # + a
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
plt.savefig(f"images/cube-face-round-iq4nl-{prefix}ary-{M}x{M}.png")
plt.close()
