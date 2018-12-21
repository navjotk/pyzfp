import numpy as np
from pyzfp import compress as compress_cy, decompress as decompress_cy
from zfp_ct import compress as compress_ct, decompress as decompress_ct
from timeit import default_timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Timer(object):
    def __init__(self, tracker):
        self.timer = default_timer
        self.tracker = tracker
        
    def __enter__(self):
        self.start = self.timer()
        return self
        
    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        self.tracker.append(self.elapsed)
        

def test_compress_decompress():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
    tolerance = 0.0000001
    compressed = compress(a, tolerance=tolerance)
    recovered = decompress(compressed, a.shape, a.dtype, tolerance=tolerance)
    assert(a.shape == recovered.shape)
    assert(np.allclose(a, recovered))

def compare_compressors():
    print("Measuring ctypes compressor")
    compress_ct_times, decompress_ct_times, sizes = measure_compressor(compress_ct, decompress_ct)
    print("Measuring cython compressor")
    compress_cy_times, decompress_cy_times, sizes = measure_compressor(compress_cy, decompress_cy)

    plt.plot(sizes, compress_ct_times, linestyle='solid', color='red', label='Compress (Ctypes)')
    plt.plot(sizes, decompress_ct_times, linestyle='dashed', color='red', label='Decompress (Ctypes)')
    plt.plot(sizes, compress_cy_times, linestyle='solid', color='blue', label='Compress (Cython)')
    plt.plot(sizes, decompress_cy_times, linestyle='dashed', color='blue', label='Decompress (Cython)')

    plt.xlabel('Array size (doubles)')
    plt.ylabel('Time (s)')
    plt.title('Comparison of calling ZFP library from Ctypes vs Cython')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    plt.savefig('ctypes_vs_cython_compression.png', bbox_inches='tight')
    

def measure_compressor(compress, decompress):
    sizes = [10, 100, 1000, 10000, 100000]#, 1000000, 10000000, 100000000] #, 1000000000, 10000000000]
    compress_timings = []
    decompress_timings = []
    tolerance = 0.0001
    for i in sizes:
        a = np.linspace(0, i, num=i)
        compress_time, compressed = measure(compress, (a, ), {'tolerance': tolerance})
        decompress_time, decompressed = measure(decompress, (compressed, a.shape, a.dtype), {'tolerance': tolerance})
        compress_timings.append(compress_time)
        decompress_timings.append(decompress_time)
        print(i)
        assert(np.allclose(a, decompressed, atol=tolerance))
    return compress_timings, decompress_timings, sizes


def measure(fn, args, kwargs, repeats=3):
    times = []
    for i in range(repeats):
        with Timer(times):
            return_value = fn(*args, **kwargs)
    return min(times), return_value


compare_compressors()
