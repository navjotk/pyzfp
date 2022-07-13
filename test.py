from functools import reduce
import operator

import pytest
import numpy as np
from pyzfp import compress, decompress


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compress_decompress(order, ndim, dtype):
    shape = []
    for i in range(ndim):
        shape.append(100 - i)

    a = np.linspace(0, 100, num=reduce(operator.mul, shape), dtype=dtype)
    a = a.reshape(shape, order=order)
    tolerance = np.finfo(dtype).resolution
    compressed = compress(a, tolerance=tolerance)
    recovered = decompress(compressed, a.shape, a.dtype,
                           tolerance=tolerance, order=order)

    compression_ratio = len(compressed) / a.nbytes
    assert compression_ratio < 1
    a.flags == recovered.flags
    assert(a.shape == recovered.shape)
    assert(np.allclose(a, recovered, atol=tolerance))

    if order == "C":
        assert recovered.flags.c_contiguous
    else:
        assert recovered.flags.f_contiguous


@pytest.mark.parametrize("order", ["C", "F"])
def test_dim_order(order):
    a = np.arange(32, dtype=np.float32).reshape((8, 4), order=order)
    compressed = compress(a, rate=8)
    recovered = decompress(compressed[0:16], (4, 4), np.dtype('float32'),
                           rate=8, order=order)
    if order == "C":
        b = np.arange(16, dtype=np.float32).reshape((4, 4), order=order)
    elif order == "F":
        b = np.array(
            [[ 0,  8, 16, 24],
             [ 1,  9, 17, 25],
             [ 2, 10, 18, 26],
             [ 3, 11, 19, 27]], 
            dtype=np.float32, order="F")
    assert(np.allclose(recovered, b))

    recovered = decompress(compressed, (8,4), np.dtype('float32'), rate=8, order=order)
    assert(np.allclose(recovered, a))
