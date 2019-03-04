# pyzfp
Quick and dirty wrapper over the [zfp compression library](https://computation.llnl.gov/projects/floating-point-compression). This is the second version, rewritten using Cython because the earlier version using ctypes was slow. [Click here](https://github.com/navjotk/pyzfp/blob/ctypes_vs_cython/ctypes_vs_cython_compression.png) for performance comparison. 

# Installation
```
make
```
This should download zfp version 0.5.3, compile it (with OPENMP
threading enabled) and leave the shared library ready-to-use  as
`zfp-0.5.3/lib/libzfp.so`. The current (hacky) implementation assumes
the presence of the shared library in this location. 

# Usage

A sample program that demonstrates the use of the library: (also contents of test.py):
```
from zfp import compress, decompress


a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))



tolerance = 0.0000001
parallel = True
compressed = compress(a, tolerance=tolerance, parallel=parallel)

recovered = decompress(compressed, a.shape, a.dtype, tolerance=tolerance)
print(len(a.tostring()))
print(len(compressed))
print(np.linalg.norm(recovered-a))
```

```
python test.py
```

