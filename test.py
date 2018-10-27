import numpy as np

from zfp import compress, decompress


a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))


tolerance = 0.0000001
compressed = compress(a, tolerance=tolerance)

recovered = decompress(compressed, a.shape, a.dtype, tolerance=tolerance)
print(len(a.tostring()))
print(len(compressed))
print(np.linalg.norm(recovered-a))
