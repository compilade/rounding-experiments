import ctypes
import numpy as np
import os
from pathlib import Path

curdir = Path(os.path.dirname(__file__))

# if not (curdir / "rounding.so").is_file():
os.system(
    f"gcc -g -O2 -shared -o '{curdir}/rounding-impl.so' -fPIC '{curdir}/rounding-impl.c'"
)

c_float_p = ctypes.POINTER(ctypes.c_float)

rounding_dll = ctypes.CDLL(str(curdir / "rounding-impl.so"))

rounding_dll.anyrize_qx.restype = None
rounding_dll.anyrize_qx.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qx(x: np.ndarray, nmax: int, qw: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qx(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmax,
    )
    return out


rounding_dll.anyrize_qxg.restype = None
rounding_dll.anyrize_qxg.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qxg(x: np.ndarray, nmax: int, qw: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qxg(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmax,
    )
    return out


rounding_dll.anyrize_qkxs.restype = None
rounding_dll.anyrize_qkxs.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
)


def anyrize_qkxs(
    x: np.ndarray,
    nmin: int,
    nmax: int,
    qw: np.ndarray | None = None,
    signed_scale: bool = True,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxs(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmin,
        nmax,
        signed_scale,
    )
    return out


rounding_dll.anyrize_q3.restype = None
rounding_dll.anyrize_q3.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_q3(x: np.ndarray, nmax: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_q3(x, out, x.shape[-1], x.shape[-2], nmax)
    return out


rounding_dll.anyrize_qp.restype = None
rounding_dll.anyrize_qp.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qp(x: np.ndarray, nmax: int, qw: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qp(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmax,
    )
    return out


rounding_dll.anyrize_qkx2_q4_k.restype = None
rounding_dll.anyrize_qkx2_q4_k.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkx2_q4_k(x: np.ndarray, nmax: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkx2_q4_k(x, out, x.shape[-1], x.shape[-2], nmax)
    return out


rounding_dll.anyrize_qkx3_q4_k.restype = None
rounding_dll.anyrize_qkx3_q4_k.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkx3_q4_k(
    x: np.ndarray, nmax: int, qw: np.ndarray | None = None
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkx3_q4_k(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmax,
    )
    return out


rounding_dll.anyrize_qkxcm_q4_k.restype = None
rounding_dll.anyrize_qkxcm_q4_k.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkxcm_q4_k(x: np.ndarray, nmax: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxcm_q4_k(x, out, x.shape[-1], x.shape[-2], nmax)
    return out


rounding_dll.anyrize_qkxh.restype = None
rounding_dll.anyrize_qkxh.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
)


def anyrize_qkxh(
    x: np.ndarray,
    nmin: int,
    nmax: int,
    qw: np.ndarray | None = None,
    signed_scale: bool = True,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxh(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmin,
        nmax,
        signed_scale,
    )
    return out


rounding_dll.anyrize_qkxsh.restype = None
rounding_dll.anyrize_qkxsh.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkxsh(
    x: np.ndarray,
    nmin: int,
    nmax: int,
    qw: np.ndarray | None = None,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxsh(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmin,
        nmax,
    )
    return out


rounding_dll.anyrize_qkxh_nl.restype = None
rounding_dll.anyrize_qkxh_nl.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C"),
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_bool,
)


kvalues_iq4nl = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

def anyrize_qkxh_nl(
    x: np.ndarray,
    kvalues: np.ndarray,
    qw: np.ndarray | None = None,
    signed_scale: bool = True,
    fast: bool = False,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxh_nl(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        kvalues.astype(np.int8, copy=False),
        kvalues.shape[0],
        signed_scale,
        fast,
    )
    return out


rounding_dll.anyrize_qkxmh.restype = None
rounding_dll.anyrize_qkxmh.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
)


def anyrize_qkxmh(
    x: np.ndarray,
    nmax: int,
    qw: np.ndarray | None = None,
    signed_min: bool = False,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxmh(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
        nmax,
        signed_min,
    )
    return out


rounding_dll.anyrize_iq4nl.restype = None
rounding_dll.anyrize_iq4nl.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_iq4nl(x: np.ndarray, qw: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_iq4nl(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
    )
    return out


rounding_dll.anyrize_iq4nl_g.restype = None
rounding_dll.anyrize_iq4nl_g.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_iq4nl_g(x: np.ndarray, qw: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_iq4nl_g(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
    )
    return out


rounding_dll.anyrize_qkxs_iq4nl.restype = None
rounding_dll.anyrize_qkxs_iq4nl.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkxs_iq4nl(x: np.ndarray, qw: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxs_iq4nl(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
    )
    return out


rounding_dll.anyrize_qkxs_iq4nl_signed.restype = None
rounding_dll.anyrize_qkxs_iq4nl_signed.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkxs_iq4nl_signed(
    x: np.ndarray, qw: np.ndarray | None = None
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxs_iq4nl_signed(
        x,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        x.shape[-1],
        x.shape[-2],
    )
    return out


rounding_dll.anyrize_qkxs_iqxnl_signed.restype = None
rounding_dll.anyrize_qkxs_iqxnl_signed.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C"),
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C"),
    c_float_p,
    np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")
    ),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


def anyrize_qkxs_iqxnl_signed(
    x: np.ndarray, kvalues: np.ndarray, qw: np.ndarray | None = None
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    kvalues = kvalues.astype(np.int8, copy=False)
    out = np.zeros_like(x)
    rounding_dll.anyrize_qkxs_iqxnl_signed(
        x,
        kvalues,
        (
            qw.astype(np.float32, copy=False).ctypes.data_as(c_float_p)
            if qw is not None
            else ctypes.cast(0, c_float_p)
        ),
        out,
        kvalues.size,
        x.shape[-1],
        x.shape[-2],
    )
    return out
