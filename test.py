import numpy as np
from pyzfp import compress, decompress


def test_compress_decompress():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
    tolerance = 0.0000001
    compressed = compress(a, tolerance=tolerance)
    recovered = decompress(compressed, a.shape, a.dtype, tolerance=tolerance)
    assert(a.shape == recovered.shape)
    assert(np.allclose(a, recovered))


def test_dim_order():
    a = np.arange(32, dtype=np.float32).reshape((8, 4))
    compressed = compress(a, rate=8)
    recovered = decompress(compressed[0:16], (4, 4), np.dtype('float32'),
                           rate=8)
    b = np.arange(16, dtype=np.float32).reshape((4, 4))
    assert(np.allclose(recovered, b))


test_compress_decompress()
test_dim_order()
