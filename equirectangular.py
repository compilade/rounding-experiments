import matplotlib.pyplot as plt
import numpy as np
import rounding_c

# Widest part of the image
# W = 1024
W = 2048


def arr_to_pixels(v: np.ndarray) -> np.ndarray:
    qw = v * v
    # qw = np.ones_like(v)

    # q = rounding_c.anyrize_qkxsh(v, -2, 2, qw)
    # q = rounding_c.anyrize_qkx2_q4_k(v, 4)
    q = rounding_c.anyrize_qkxcm_q4_k(v, 4)
    # q = rounding_c.anyrize_qkxs_iq4nl_signed(v, qw)
    # q = rounding_c.anyrize_iq4nl(v, qw)
    # q = rounding_c.anyrize_q3(v, 4)
    # q = rounding_c.anyrize_qkxs(v, -4, 3, qw)

    return np.sum(qw * v * q, axis=-1, keepdims=True) / np.sqrt(
        np.sum(qw * q * q, axis=-1, keepdims=True) * np.sum(qw * v * v, axis=-1, keepdims=True)
    )


theta = np.linspace(0, 2 * np.pi, W)
phi = np.linspace(0, np.pi, W // 2)


plane = np.array(
    [
        [np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]
        for p in phi
        for t in theta
    ]
)

cos = arr_to_pixels(plane).reshape((W // 2, W))

print(f"{np.min(cos)=}")
print(f"{np.mean(cos)=}")
print(f"{np.max(cos)=}")

plt.figure()
plt.imshow(cos)
plt.show()

plt.figure(dpi=96, figsize=(cos.shape[-1] / 96, cos.shape[-2] / 96))
plt.figimage(cos)
plt.savefig(f"images/equirectangular-qkx2-4-{W}.png")
plt.close()
