import numpy as np
from pyzfp import compress, decompress


def test_compress_decompress():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
    tolerance = 0.0000001
    compressed = compress(a, tolerance=tolerance)
    recovered = decompress(compressed, a.shape, a.dtype, tolerance=tolerance)
    assert(a.shape == recovered.shape)
    assert(np.allclose(a, recovered))


test_compress_decompress()
