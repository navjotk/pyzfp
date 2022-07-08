import pytest
import numpy as np
from pyzfp import compress, decompress


@pytest.mark.parametrize("order", ["C", "F"])
def test_compress_decompress(order):
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100), order=order)
    tolerance = 0.0000001
    compressed = compress(a, tolerance=tolerance)
    recovered = decompress(compressed, a.shape, a.dtype,
                           tolerance=tolerance, order=order)
    a.flags == recovered.flags
    assert(a.shape == recovered.shape)
    assert(np.allclose(a, recovered))


@pytest.mark.parametrize("order", ["C", "F"])
def test_dim_order(order):
    a = np.arange(32, dtype=np.float32).reshape((8, 4), order=order)
    compressed = compress(a, rate=8)
    recovered = decompress(compressed[0:16], (4, 4), np.dtype('float32'),
                           rate=8, order=order)
    b = np.arange(16, dtype=np.float32).reshape((4, 4), order=order)
    assert(np.allclose(recovered, b))


test_compress_decompress('C')
test_compress_decompress('F')
test_dim_order('C')
test_dim_order('F')
