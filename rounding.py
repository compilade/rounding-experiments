from typing import Literal
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class QuantInfo:
    v: np.ndarray
    q: np.ndarray
    sc: float | np.float64 | np.ndarray
    mn: np.ndarray | None = None
    iscales: np.ndarray | None = None
    angles: np.ndarray | None = None


# round away from zero
# ref: https://stackoverflow.com/a/59143326/22827863
def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


def anyrize_inv_sq(a: np.ndarray, min_max: int, axis: Literal[-1] | None = None):
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    # sort?
    # find the corresponding scales <--
    # wait, there's 2 things... the scales and the corresponding values from a
    # The rounding scales are no longer always sort(1 / (2*abs(a))),
    # The rounding scales are sort([1,3,5,7,9, ...] / (2*abs(a)))
    # How to try them in order??? (sort ascending???)
    # Wait, the angles are the cumsums of the descending inverse scales????
    # The inverse rounding scales are sort_desc((2 * abs(a)) / [1,3,5,7,9,...])
    # To try the inv scales in order, sort them descending
    # The angles are proportional to (cumsum(numer(-sort(-iscales)))**2) / cumsum(denom(-sort(-iscales)))
    # find the best
    # round?
    # Okay, let's implement that.
    a = a.astype(np.float32, copy=False)
    shape = a.shape
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    odd = np.array([1 + (2 * i) for i in range(min_max)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(a[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    iscales = abs(ab) / odd
    ids = np.argsort(-iscales, axis=axis)
    sa = np.take_along_axis(ab, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    c = np.cumsum(abs(sa), axis=axis)
    cn = (c * c) / np.cumsum(so, axis=axis)

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    iscale = 2 * np.take_along_axis(iscales, mid, axis=axis)

    q = np.clip(np_roundf(a / iscale), -abs(min_max), abs(min_max))

    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )

    sis = np.take_along_axis(iscales, ids, axis=axis)

    # print(q * sc)

    return QuantInfo(
        v=q * sc,
        iscales=sis,
        angles=np.sqrt(cn / np.sum(a * a, axis=axis, keepdims=(axis is not None))),
        q=q,
        sc=sc,
    )


def anyrize_inv_sqrt(a: np.ndarray, min_max: int, axis: Literal[-1] | None = None):
    a = a.astype(np.float32, copy=False)
    shape = a.shape
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    odd = np.array([1 + (2 * i) for i in range(min_max)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(a[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    iscales = abs(ab) / odd
    ids = np.argsort(-iscales, axis=axis)
    sa = np.take_along_axis(ab, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    c = np.cumsum(abs(sa), axis=axis)
    cn = c / np.sqrt(np.cumsum(so, axis=axis))

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    iscale = 2 * np.take_along_axis(iscales, mid, axis=axis)

    q = np.clip(np_roundf(a / iscale), -abs(min_max), abs(min_max))

    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )

    sis = np.take_along_axis(iscales, ids, axis=axis)

    return QuantInfo(
        v=q * sc,
        iscales=sis,
        angles=cn / np.sum(a * a, axis=axis, keepdims=(axis is not None)),
        q=q,
        sc=sc,
    )


def anyrize_sq(a: np.ndarray, min_max: int, axis: Literal[-1] | None = None):
    a = a.astype(np.float32, copy=False)
    shape = a.shape
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    odd = np.array([1 + (2 * i) for i in range(min_max)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(a[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    scales = odd / abs(ab)
    ids = np.argsort(scales, axis=axis)
    sa = np.take_along_axis(ab, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    c = np.cumsum(abs(sa), axis=axis)
    cn = (c * c) / np.cumsum(so, axis=axis)

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    scale = np.take_along_axis(scales, mid, axis=axis) / 2

    q = np.clip(np_roundf(a * scale), -abs(min_max), abs(min_max))

    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )

    sis = 1 / np.take_along_axis(scales, ids, axis=axis)

    return QuantInfo(
        v=q * sc,
        iscales=sis,
        angles=np.sqrt(cn / np.sum(a * a, axis=axis, keepdims=(axis is not None))),
        q=q,
        sc=sc,
    )


def anyrize_sqrt(a: np.ndarray, min_max: int, axis: Literal[-1] | None = None):
    a = a.astype(np.float32, copy=False)
    shape = a.shape
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    odd = np.array([1 + (2 * i) for i in range(min_max)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(a[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    scales = odd / abs(ab)
    ids = np.argsort(scales, axis=axis)
    sa = np.take_along_axis(ab, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    c = np.cumsum(abs(sa), axis=axis)
    cn = c / np.sqrt(np.cumsum(so, axis=axis))

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    scale = np.take_along_axis(scales, mid, axis=axis) / 2

    q = np.clip(np_roundf(a * scale), -abs(min_max), abs(min_max))

    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )

    sis = 1 / np.take_along_axis(scales, ids, axis=axis)

    return QuantInfo(
        v=q * sc,
        iscales=sis,
        angles=cn / np.sum(a * a, axis=axis, keepdims=(axis is not None)),
        q=q,
        sc=sc,
    )


def anyrize_offset_mean(
    a: np.ndarray, min_max: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    # Two steps which minimize the squared difference
    # One step would be the squared median
    # Wait... it's the k-medians we're searching....
    # But it's *also* the same as rounding in the first quadrant!!!
    # (by first assuming a min = np.min(v))
    # (but that doesn't seem like the ideal...)
    # But is it really? Maybe not? Need a proof!
    # Do both directions need to be tried?
    a = a.astype(np.float32, copy=False)
    off = a - np.mean(a, axis=axis, keepdims=True)
    shape = a.shape
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    odd = np.array([1 + (2 * i) for i in range(min_max)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(off[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    iscales = abs(ab) / odd
    ids = np.argsort(-iscales, axis=axis)
    sa = np.take_along_axis(ab, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    c = np.cumsum(abs(sa), axis=axis)
    cn = (c * c) / np.cumsum(so, axis=axis)

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    iscale = 2 * np.take_along_axis(iscales, mid, axis=axis)

    q = np.clip(np_roundf(off / iscale), -abs(min_max), abs(min_max))

    # The scale is the correction on the plane between q and [1,1,1,...]
    # to the perpendicular (to [1,1,1,...]) projection of q compared to a.
    # Apparently, projecting q on [1,1,1,...] is the same as taking its mean!!
    centered = q - np.mean(q, axis=axis, keepdims=(axis is not None))
    # FIXME: This isn't always the best scale
    with np.errstate(divide="ignore"):
        sc = np.where(
            centered != 0,
            np.sum(centered * a, axis=axis, keepdims=(axis is not None))
            / np.sum(centered * centered, axis=axis, keepdims=(axis is not None)),
            0,
        )

    # The min can rotate the vector on the plane between q and [1,1,1,...]
    # The cosine with the original a needs to be maximal.
    # Which means we need to find the closest point on the plane?
    # What is the min in that coordinate system?
    mn = sc * np.mean(q, axis=axis, keepdims=(axis is not None)) - np.mean(
        a, axis=axis, keepdims=(axis is not None)
    )

    sis = np.take_along_axis(iscales, ids, axis=axis)

    # print(q * sc - mn)
    return QuantInfo(
        v=q * sc - mn,
        iscales=sis,
        angles=np.sqrt(cn / np.sum(a * a, axis=axis, keepdims=(axis is not None))),
        q=q,
        sc=sc,
        mn=mn,
    )


def anyrize_offset_min(
    a: np.ndarray, nmax: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    # Two steps which minimize the squared difference
    # One step would be the squared median
    # Wait... it's the k-medians we're searching....
    # But it's *also* the same as rounding in the first quadrant!!!
    # (by first assuming a min = np.min(v))
    # (but that doesn't seem like the ideal...)
    # But is it really? Maybe not? Need a proof!
    # Do both directions need to be tried?
    a = a.astype(np.float32, copy=False)
    off = a - np.min(a, axis=axis, keepdims=True)
    shape = a.shape
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    odd = np.array([1 + (2 * i) for i in range(nmax)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(off[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    iscales = abs(ab) / odd
    ids = np.argsort(-iscales, axis=axis)
    sa = np.take_along_axis(ab, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    c = np.cumsum(abs(sa), axis=axis)
    cn = (c * c) / np.cumsum(so, axis=axis)

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    iscale = 2 * np.take_along_axis(iscales, mid, axis=axis)

    q = np.clip(np_roundf(off / iscale), 0, abs(nmax))

    # The scale is the correction on the plane between q and [1,1,1,...]
    # to the perpendicular (to [1,1,1,...]) projection of q compared to a.
    # Apparently, projecting q on [1,1,1,...] is the same as taking its mean!!
    centered = q - np.mean(q, axis=axis, keepdims=(axis is not None))
    # FIXME: This isn't always the best scale
    with np.errstate(divide="ignore"):
        sc = np.where(
            centered != 0,
            np.sum(centered * a, axis=axis, keepdims=(axis is not None))
            / np.sum(centered * centered, axis=axis, keepdims=(axis is not None)),
            0,
        )

    # The min can rotate the vector on the plane between q and [1,1,1,...]
    # The cosine with the original a needs to be maximal.
    # Which means we need to find the closest point on the plane?
    # What is the min in that coordinate system?
    mn = sc * np.mean(q, axis=axis, keepdims=(axis is not None)) - np.mean(
        a, axis=axis, keepdims=(axis is not None)
    )

    sis = np.take_along_axis(iscales, ids, axis=axis)

    # print(q * sc - mn)
    return QuantInfo(
        v=q * sc - mn,
        iscales=sis,
        angles=np.sqrt(cn / np.sum(a * a, axis=axis, keepdims=(axis is not None))),
        q=q,
        sc=sc,
        mn=mn,
    )


def anyrize_offset_min_mean(
    a: np.ndarray, nmax: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    # Two steps which minimize the squared difference
    # One step would be the squared median
    # Wait... it's the k-medians we're searching....
    # But it's *also* the same as rounding in the first quadrant!!!
    # (by first assuming a min = np.min(v))
    # (but that doesn't seem like the ideal...)
    # But is it really? Maybe not? Need a proof!
    # Do both directions need to be tried?
    a = a.astype(np.float32, copy=False)
    N = a.size if axis is None else a.shape[axis]
    off = np.min(a, axis=axis, keepdims=True)
    mea = np.mean(a, axis=axis, keepdims=True)
    shape = a.shape
    odd = np.array([2 * i + 1 for i in range(nmax)], dtype=np.float32)
    # TODO: does this only work for axis=-1 | None?
    ab, odd = np.broadcast_arrays(a[..., np.newaxis], odd)
    ab = ab.reshape((*shape[:-1], -1))
    odd = odd.reshape((*shape[:-1], -1))

    # TODO: handle assymmetric quantization by making "odd" apply differently to positive and negative values
    # TODO(research): heuristic for side with more precision?
    # All the .5 --> 0.5 * (1, 3, 5, 7, 9,)
    iscales = (ab - off) / odd
    ids = np.argsort(-iscales, axis=axis)
    sa = np.take_along_axis(ab - mea, ids, axis=axis)
    so = np.take_along_axis(odd, ids, axis=axis)

    # Project the quantized vector on the hyperplane normal to [1,1,1,...]
    # and then calculate the squared cosine of the angle
    c = np.cumsum(sa, axis=axis) - (
        np.sum(sa, axis=axis, keepdims=True)
        * np.cumsum(np.ones_like(sa), axis=axis)
        / N
    )
    norms = np.cumsum(so, axis=axis) - (
        np.square(np.cumsum(np.ones_like(so), axis=axis)) / N
    )
    with np.errstate(divide="ignore"):
        cn = np.where(norms != 0, np.square(c) / norms, 0)

    # FIXME: Need the last max to avoid recalculating the scale later
    mid = np.take_along_axis(ids, np.argmax(cn, axis=axis, keepdims=True), axis=axis)

    iscale = 2 * np.take_along_axis(iscales, mid, axis=axis)

    q = np.clip(np_roundf((a - off) / iscale), 0, abs(nmax))

    # The scale is the correction on the plane between q and [1,1,1,...]
    # to the perpendicular (to [1,1,1,...]) projection of q compared to a.
    # Apparently, projecting q on [1,1,1,...] is the same as taking its mean!!
    centered = q - np.mean(q, axis=axis, keepdims=(axis is not None))
    # FIXME: This isn't always the best scale
    with np.errstate(divide="ignore"):
        sc = np.where(
            centered != 0,
            np.sum(centered * a, axis=axis, keepdims=(axis is not None))
            / np.sum(centered * centered, axis=axis, keepdims=(axis is not None)),
            0,
        )

    # The min can rotate the vector on the plane between q and [1,1,1,...]
    # The cosine with the original a needs to be maximal.
    # Which means we need to find the closest point on the plane?
    # What is the min in that coordinate system?
    mn = sc * np.mean(q, axis=axis, keepdims=(axis is not None)) - np.mean(
        a, axis=axis, keepdims=(axis is not None)
    )

    sis = np.take_along_axis(iscales, ids, axis=axis)

    # print(q * sc - mn)
    return QuantInfo(
        v=q * sc - mn,
        iscales=sis,
        angles=np.sqrt(
            cn / np.sum(np.square(a - mea), axis=axis, keepdims=(axis is not None))
        ),
        q=q,
        sc=sc,
        mn=mn,
    )


def binary(a: np.ndarray, axis: Literal[-1] | None = None) -> QuantInfo:
    q = np.where(a > 0, 1, -1)
    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )
    return QuantInfo(v=q * sc, q=q, sc=sc)


def binary_offset(a: np.ndarray, axis: Literal[-1] | None = None) -> QuantInfo:
    N = a.size if axis is None else a.shape[axis]
    am = a - np.mean(a, axis=axis, keepdims=True)
    s = np.sort(am, axis=axis)
    # Try each min (is that even possible??)
    # (yes because it's each min which changes the rouding that is relevant,
    #  so for binary it's each one which changes a sign)
    ss = np.sum(s, axis=axis, keepdims=True)
    c = np.cumsum(s, axis=axis)
    # Now we need to calculate the cos(angle) on the unit hyperplane (or something proportional)
    dot = ss - 2 * c  # progressively bigger min
    # Offset norm
    # sum(1*1) - sum(1)**2 / N
    norms = N - (np.square(N - 2 * np.cumsum(np.ones_like(s), axis=axis)) / N)
    with np.errstate(divide="ignore"):
        cos = np.where(norms != 0, np.square(dot) / (norms), 0)

    i = np.argmax(cos, axis=axis, keepdims=True)

    m = np.take_along_axis(s, i, axis=axis)

    q = np.where(am <= m, -1, 1)

    # The scale is the correction on the plane between q and [1,1,1,...]
    # to the perpendicular (to [1,1,1,...]) projection of q compared to a.
    # Apparently, projecting q on [1,1,1,...] is the same as taking its mean!!
    centered = q - np.mean(q, axis=axis, keepdims=(axis is not None))
    # FIXME: This isn't always the best scale
    with np.errstate(divide="ignore"):
        sc = np.where(
            centered != 0,
            np.sum(centered * a, axis=axis, keepdims=(axis is not None))
            / np.sum(centered * centered, axis=axis, keepdims=(axis is not None)),
            0,
        )

    # The min can rotate the vector on the plane between q and [1,1,1,...]
    # The cosine with the original a needs to be maximal.
    # Which means we need to find the closest point on the plane?
    # What is the min in that coordinate system?
    mn = sc * np.mean(q, axis=axis, keepdims=(axis is not None)) - np.mean(
        a, axis=axis, keepdims=(axis is not None)
    )

    # FIXME: the angle is wrong here
    return QuantInfo(v=q * sc - mn, q=q, sc=sc, mn=mn, angles=np.sqrt(cos))


def absmax_round(
    a: np.ndarray, min_max: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    q = np.clip(
        np_roundf(abs(min_max) * a / np.max(np.abs(a), axis=axis, keepdims=True)),
        -abs(min_max),
        abs(min_max),
    )
    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )
    return QuantInfo(v=q * sc, q=q, sc=sc)


def absmax_dumb_round(
    a: np.ndarray, min_max: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    q = np.clip(
        np_roundf(abs(min_max) * a / np.max(np.abs(a), axis=axis, keepdims=True)),
        -abs(min_max),
        abs(min_max),
    )
    sc = np.max(np.abs(a), axis=axis, keepdims=(axis is not None)) / abs(min_max)
    # sc = abs(min_max) / np.max(np.abs(a))
    return QuantInfo(v=q * sc, q=q, sc=sc)


def offset_dumb_round(
    a: np.ndarray, min_max: int, axis: Literal[-1] | None = None
) -> QuantInfo:

    mn = np.min(a, axis=axis, keepdims=True)
    sc = np.max(a - mn, axis=axis, keepdims=True) / abs(min_max * 2)

    q = np.clip(
        np_roundf((a - mn) / sc),
        0,
        abs(min_max * 2),
    )
    return QuantInfo(v=q * sc + mn, q=q, sc=sc, mn=mn)


def absmean_round(
    a: np.ndarray, min_max: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    q = np.clip(
        np_roundf(abs(min_max) * a / np.mean(np.abs(a), axis=axis, keepdims=True)),
        -abs(min_max),
        abs(min_max),
    )
    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )
    return QuantInfo(v=q * sc, q=q, sc=sc)


def absmedian_round(
    a: np.ndarray, min_max: int, axis: Literal[-1] | None = None
) -> QuantInfo:
    q = np.clip(
        np_roundf(abs(min_max) * a / np.median(np.abs(a), axis=axis, keepdims=True)),
        -abs(min_max),
        abs(min_max),
    )
    sc = np.sum(q * a, axis=axis, keepdims=(axis is not None)) / np.sum(
        q * q, axis=axis, keepdims=(axis is not None)
    )
    return QuantInfo(v=q * sc, q=q, sc=sc)


def show(n: str, q: QuantInfo, a: np.ndarray):
    mse = np.sum(np.square(q.v - a).ravel())
    cos = np.dot(
        q.v.ravel() / np.sqrt(q.v.ravel().dot(q.v.ravel())),
        a.ravel() / np.sqrt(a.ravel().dot(a.ravel())),
    )
    angle = np.arccos(cos) * 180 / np.pi
    print(f"{n:<16} {mse:>16.8f}  {angle:>11.8f}")
    # print(
    #     n + "_wiggle",
    #     np.sum(np.square((q.v + 0.1) - a).ravel()),
    #     np.sum(np.square((q.v - 0.1) - a).ravel()),
    # )


rng = np.random.default_rng(42)

for i in range(8):
    a = rng.laplace(
        size=(
            1,
            32,
        )
    )

    min_max = 7
    axis = -1

    print(a)
    show("inv_sq", anyrize_inv_sq(a, min_max, axis=axis), a)
    show("inv_sqrt", anyrize_inv_sqrt(a, min_max, axis=axis), a)
    show("sq", anyrize_sq(a, min_max, axis=axis), a)
    show("sqrt", anyrize_sqrt(a, min_max, axis=axis), a)
    show("offset_mean", anyrize_offset_mean(a, min_max, axis=axis), a)
    show("offset_min", anyrize_offset_min(a, 2 * min_max, axis=axis), a)
    show("offset_min_mean", anyrize_offset_min_mean(a, 2 * min_max, axis=axis), a)
    show("offset_dumb", offset_dumb_round(a, min_max, axis=axis), a)
    show("absmax", absmax_round(a, min_max, axis=axis), a)
    show("absmax_dumb", absmax_dumb_round(a, min_max, axis=axis), a)
    show("absmean", absmean_round(a, min_max, axis=axis), a)
    show("absmedian", absmedian_round(a, min_max, axis=axis), a)
    show("binary", binary(a, axis=axis), a)
    show("bin_offset", binary_offset(a, axis=axis), a)
    print("----")