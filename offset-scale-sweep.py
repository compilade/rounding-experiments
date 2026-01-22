import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable

# W = 4096
W = 2048
H = W // 2

B = 5
N = 8

def print_dict(d):
    print("{")
    for key, val in d.items():
        print(f"  {key}: {val},")
    print("}")


def partial_vec(current: tuple[int, ...], max: int, remaining: int) -> Iterable[tuple[int, ...]]:
    if remaining <= 0:
        yield current
        return
    for i in range(max + 1):
        yield from partial_vec(current + (i,), i, remaining - 1)

# simplify symmetries by ordering the components
# of all the representable vectors
def all_vecs_ordered(base: int, dims: int) -> np.ndarray:
    acc = tuple(v + (0,) for v in partial_vec((), base - 1, dims - 1))
    # for i in range(dims):
    #     for j in range(base):
    #         for k in range():
    return np.array(acc)

def anyrize_offset_ordered_orig(v: np.ndarray, base: int = 2) -> np.ndarray:
    a = all_vecs_ordered(base, v.shape[-1])
    m = a - a.mean(axis=-1, keepdims=True)
    vi = np.argsort(-v, axis=-1, stable=True)
    vs = np.take_along_axis(v, vi, axis=-1)

    sumlx = m[None, :, :] @ vs[:, :, None]
    sumlx = sumlx.reshape(v.shape[0], m.shape[0])
    proj2 = np.square(sumlx)
    suml2 = m[:, None, :] @ m[:, :, None]
    suml2 = suml2.reshape(1, m.shape[0])

    with np.errstate(divide="ignore"):
        sim = np.where(suml2 > 0, proj2 / suml2, 0)
        sim = np.where(sumlx > 0, sim, 0)

    qi = np.argmax(sim, axis=-1, keepdims=True)
    qb = np.take_along_axis(a[None, :, :], qi[:, :, None], axis=1).reshape(
        v.shape[0], a.shape[-1]
    )
    return qb

rng = np.random.default_rng()
v = rng.laplace(size=(N,)).reshape(1, 1, -1)
# v = np.array([2.1987305, 1.0421944, 0.7682289, 0.24435928, 0.12848878, -0.24960263, -0.46659088, -4.641285]).reshape(1, 1, -1)
print(v)

# we want the min value to be offset to range -0.5..0.5 (ideally both exclusive),
# and the max should range between 0.5..(B-0.5)
mint = np.linspace(-0.5, 0.5, H, endpoint=False).reshape(-1, 1, 1)
maxt = np.linspace(0.5, B-0.5, W).reshape(1, -1, 1)

minv = v.min(axis=-1, keepdims=True)
maxv = v.max(axis=-1, keepdims=True)

# make the range match and offset considering the scale
scale = (maxt - mint) / (maxv - minv)
offset = minv * scale - mint
c = v * scale - offset

# print(c)

q = np.clip(np.round(c), 0, B - 1)

qm = q - q.mean(axis=-1, keepdims=True)
vm = v - v.mean(axis=-1, keepdims=True)
projm = vm[..., None, :] @ qm[..., :, None]
normqm = qm[..., None, :] @ qm[..., :, None]
normvm = vm[..., None, :] @ vm[..., :, None]
normm = normqm * normvm

cos = (np.square(projm) / np.where(normm > 0, normm, 0)).reshape(H, W)

# enumerate quantized vectors
quants = {}
ql = q.reshape(-1, q.shape[-1]).astype(np.int8).tolist()
cosl = cos.reshape(-1).tolist()
for row, err in zip(ql, cosl):
    t = tuple(row)
    quants[t] = err

print_dict(quants)
print(len(quants))

print(anyrize_offset_ordered_orig(v.reshape(-1, v.shape[-1]), B))

plt.figure()
plt.imshow(cos)
plt.show()

plt.figure(dpi=96, figsize=(cos.shape[-1] / 96, cos.shape[-2] / 96))
plt.figimage(cos)
plt.savefig(f"images/offset-scale-sweep-1-{W}x{H}.png")
plt.close()




