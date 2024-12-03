#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Plain max cosine calculation for N-ary rounding on a cube,
# which also takes into account all possible scales,
# and so the resulting shape is what a good rounding function should converge towards.

# [-N .. N]
N = 4

corners = np.array(
    [
        [1, i / n, j / n]
        for n in range(1, N + 1)
        for i in range(n + 1)
        for j in range(n + 1)
    ]
)
corners = corners / np.sqrt(np.sum(np.square(corners), axis=-1, keepdims=True))
corners = corners.T

# pixels
M = 256

middle = np.array([[1, i / (M - 1), j / (M - 1)] for i in range(M) for j in range(M)])
middle = middle / np.sqrt(np.sum(np.square(middle), axis=-1, keepdims=True))

print(corners.shape)
print(middle.shape)
cos = middle @ corners
cos = np.max(cos, axis=-1, keepdims=True)
print(cos.shape)
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
}.get(N, f"{N}_")

plt.figure(dpi=96, figsize=(face.shape[-1] / 96, face.shape[-2] / 96))
plt.figimage(face)
plt.savefig(f"images/cube-face-{prefix}ary-{2*M}x{2*M}.png")
plt.close()
