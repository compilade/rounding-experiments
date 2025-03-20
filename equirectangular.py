import matplotlib.pyplot as plt
import numpy as np
import rounding_c
import rounding

# Widest part of the image
# W = 1024
W = 2048
# W = 4096
# W = 8192


def arr_to_pixels(v: np.ndarray) -> np.ndarray:
    # qw = v * v
    qw = np.ones_like(v)

    # q = rounding_c.anyrize_qkxh(v, -7, 7, qw)
    # q = rounding_c.anyrize_qkx2_q4_k(v, 4)
    q = rounding_c.anyrize_qkx3_q4_k(v, 15, qw)
    # q = rounding_c.anyrize_qx(v, 8, qw)
    # q = rounding_c.anyrize_qkxcm_q4_k(v, 4)
    # q = rounding_c.anyrize_qkxs_iq4nl_signed(v, qw)
    # q = rounding_c.anyrize_iq4nl(v, qw)
    # q = rounding_c.anyrize_q3(v, 4)
    # q = rounding_c.anyrize_qkxs(v, -4, 3, qw)

    return np.sum(qw * v * q, axis=-1, keepdims=True) / np.sqrt(
        np.sum(qw * q * q, axis=-1, keepdims=True)
        * np.sum(qw * v * v, axis=-1, keepdims=True)
    )


theta = np.linspace(0, 2 * np.pi, W, endpoint=False).reshape((1, -1))
phi = np.linspace(0, np.pi, W // 2, endpoint=True).reshape((-1, 1))

cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)

coords = [
    (sin_phi @ cos_theta),
    (sin_phi @ sin_theta),
    (cos_phi @ np.ones_like(theta)),
    # (sin_phi @ sin_theta),
    # (sin_phi @ sin_theta),
    # (sin_phi @ sin_theta),
    # (sin_phi @ sin_theta),
]
## Squished, but keeps the mean at zero
# coords += [-(coords[0] + coords[1] + coords[2])]
## Rhombic dodecahedron!
# coords += [-(1/3)*(coords[0] + coords[1] + coords[2])]

equirectangle = np.concatenate([c.reshape((-1, 1)) for c in coords], axis=-1)

cos = arr_to_pixels(equirectangle).reshape((W // 2, W))

print(f"{np.min(cos)=}")
print(f"{np.mean(cos)=}")
print(f"{np.max(cos)=}")

plt.figure()
plt.imshow(cos)
plt.show()

plt.figure(dpi=96, figsize=(cos.shape[-1] / 96, cos.shape[-2] / 96))
plt.figimage(cos)
plt.savefig(f"images/equirectangular-tmp-{W}.png")
plt.close()
