from cython cimport view
import numpy as np
cimport numpy as np

cdef extern from "bitstream.h":
  cdef struct bitstream:
    pass

  cdef bitstream* stream_open(void* buffer, size_t bytes);

  cdef void stream_close(bitstream* stream);

cdef extern from "zfp/types.h":
  ctypedef unsigned int uint

cdef extern from "zfp.h":
  cdef double zfp_stream_set_accuracy(zfp_stream* stream, double tolerance);
  ctypedef enum zfp_exec_policy:
    zfp_exec_serial = 0, #/* serial execution (default) */
    zfp_exec_omp    = 1, #/* OpenMP multi-threaded execution */
    zfp_exec_cuda   = 2  #/* CUDA parallel execution */

  ctypedef enum zfp_type:
    zfp_type_none   = 0, #/* unspecified type */
    zfp_type_int32  = 1, #/* 32-bit signed integer */
    zfp_type_int64  = 2, #/* 64-bit signed integer */
    zfp_type_float  = 3, #/* single precision floating point */
    zfp_type_double = 4  #/* double precision floating point */

  #/* OpenMP execution parameters */
  ctypedef struct zfp_exec_params_omp:
    uint threads;    #/* number of requested threads */
    uint chunk_size; #/* number of blocks per chunk (1D only) */

  #/* execution parameters */
  ctypedef union zfp_exec_params:
    zfp_exec_params_omp omp; #/* OpenMP parameters */

  ctypedef struct zfp_execution:
    zfp_exec_policy policy; #/* execution policy (serial, omp, ...) */
    zfp_exec_params params; #/* execution parameters */

  ctypedef struct zfp_stream:
    pass
    #uint minbits;      # /* minimum number of bits to store per block */
    #uint maxbits;      # /* maximum number of bits to store per block */
    #uint maxprec;      # /* maximum number of bit planes to store */
    #int minexp;        # /* minimum floating point bit plane number to store */
    #bitstream* stream; # /* compressed bit stream */
    #zfp_execution exec1;# /* execution policy and parameters */

  ctypedef struct zfp_field:
      zfp_type type;       #/* scalar type (e.g. int32, double) */
      uint nx, ny, nz, nw; #/* sizes (zero for unused dimensions) */
      int sx, sy, sz, sw;  #/* strides (zero for contiguous array a[nw][nz][ny][nx]) */
      void* data;          #/* pointer to array data */

  cdef uint zfp_stream_set_precision(zfp_stream* stream, uint precision);

  cdef double zfp_stream_set_rate(
      zfp_stream* stream, #/* compressed stream */
      double rate,        #/* desired rate in compressed bits/scalar */
      zfp_type stype,      #/* scalar type to compress */
      uint dims,          #/* array dimensionality (1, 2, or 3) */
      int wra             #/* nonzero if write random access is needed */
  );

  cdef size_t zfp_stream_maximum_size(
      const zfp_stream* stream, #/* compressed stream */
      const zfp_field* field    #/* array to compress */
  );

  cdef void zfp_stream_set_bit_stream(
      zfp_stream* stream, #/* compressed stream */
      bitstream* bs       #/* bit stream to read from and write to */
  );

  cdef void zfp_stream_rewind(
      zfp_stream* stream #/* compressed bit stream */
  );

  cdef zfp_stream* zfp_stream_open(
      bitstream* stream #/* bit stream to read from and write to (may be NULL) */
  );

  cdef void zfp_stream_close(zfp_stream* stream);

  cdef int zfp_stream_set_execution(
      zfp_stream* stream,    #/* compressed stream */
      zfp_exec_policy policy #/* execution policy */
  );

  cdef zfp_field* zfp_field_1d(
    void* pointer, #/* pointer to uncompressed scalars (may be NULL) */
    zfp_type type, #/* scalar type */
    uint nx        #/* number of scalars */
   );

  cdef zfp_field* zfp_field_2d(
    void* pointer, #/* pointer to uncompressed scalars (may be NULL) */
    zfp_type type, #/* scalar type */
    uint nx,       #/* number of scalars in x dimension */
    uint ny        #/* number of scalars in y dimension */
   );

  cdef zfp_field* zfp_field_3d(
    void* pointer, #/* pointer to uncompressed scalars (may be NULL) */
    zfp_type type, #/* scalar type */
    uint nx,       #/* number of scalars in x dimension */
    uint ny,       #/* number of scalars in y dimension */
    uint nz        #/* number of scalars in z dimension */
   );

  cdef zfp_field* zfp_field_4d(
    void* pointer, #/* pointer to uncompressed scalars (may be NULL) */
    zfp_type type, #/* scalar type */
    uint nx,       #/* number of scalars in x dimension */
    uint ny,       #/* number of scalars in y dimension */
    uint nz,       #/* number of scalars in z dimension */
    uint nw        #/* number of scalars in w dimension */
   );

  cdef size_t zfp_compress(
      zfp_stream* stream,    #/* compressed stream */
      const zfp_field* field #/* field metadata */
  );

  cdef size_t zfp_decompress(
      zfp_stream* stream, #/* compressed stream */
      zfp_field* field    #/* field metadata */
  );

  cdef void zfp_field_free(
      zfp_field* field #/* field metadata */
  );

cdef void* raw_pointer_double(arr) except NULL:
    assert(arr.dtype==np.float64)
    assert(arr.flags.c_contiguous) # if this isn't true, ravel will make a copy
    cdef double[::1] mview = arr.ravel()
    return <void*>&mview[0]

cdef void* raw_pointer_float(arr) except NULL:
    assert(arr.flags.c_contiguous) # if this isn't true, ravel will make a copy
    assert(arr.dtype == np.float32)
    cdef float[::1] mview = arr.ravel()
    return <void*>&mview[0]

cdef void* raw_pointer(arr):
    if arr.dtype == np.float32:
        return raw_pointer_float(arr)
    else:
        return raw_pointer_double(arr)

zfp_types = {np.dtype('float32'): zfp_type_float, np.dtype('float64'): zfp_type_double}

cdef zfp_field* init_field(np.ndarray indata):
    data_type = zfp_types[indata.dtype]
    numdims = len((<object> indata).shape)
    shape = indata.shape
    if numdims == 1:
        return zfp_field_1d(raw_pointer(indata), data_type, shape[0])
    elif numdims == 2:
        return zfp_field_2d(raw_pointer(indata), data_type, shape[0], shape[1])
    elif numdims == 3:
        return zfp_field_3d(raw_pointer(indata), data_type, shape[0], shape[1], shape[2])
    #elif numdims == 4:
    #    return zfp_field_4d(raw_pointer(indata), data_type, shape[0], shape[1], shape[2], shape[4])
    else:
        raise ValueError("Invalid number of dimensions (valid: 1-4)")

def compress(indata, tolerance=None, precision=None, rate=None, parallel=True):
    assert(tolerance or precision or rate)
    assert(not(tolerance is not None and precision is not None))
    assert(not(tolerance is not None and rate is not None))
    assert(not(rate is not None and precision is not None))

    shape = list(reversed(indata.shape))

    field = init_field(indata)
    stream = zfp_stream_open(NULL)
    data_type = zfp_types[indata.dtype]
    # TODO Use return value to ensure we succeeded in all these 'set' calls
    if tolerance is not None:
        zfp_stream_set_accuracy(stream, tolerance)
    elif precision is not None:
        zfp_stream_set_precision(stream, precision)
    elif rate is not None:
        zfp_stream_set_rate(stream, rate, data_type, len(shape), 0)
    # Try multithreaded
    if(parallel):
        zfp_stream_set_execution(stream, zfp_exec_omp)
    bufsize = zfp_stream_maximum_size(stream, field)
    cdef char[::1] buff = view.array(shape=(bufsize,), itemsize=sizeof(char), format='B')
    bitstream = stream_open(<void *>&buff[0], bufsize)
    zfp_stream_set_bit_stream(stream, bitstream)
    zfp_stream_rewind(stream)
    zfpsize = zfp_compress(stream, field)

    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bitstream)
    return buff[:zfpsize]


def decompress(char[::1] compressed, shape, dtype, tolerance=None, precision=None,
               rate=None, parallel=True):
    assert(tolerance or precision or rate)
    assert(not(tolerance is not None and precision is not None))
    assert(not(tolerance is not None and rate is not None))
    assert(not(rate is not None and precision is not None))
    outdata = np.zeros(shape, dtype=dtype)
    data_type = zfp_types[dtype]
    shape = list(reversed(shape))
    field = init_field(outdata)
    stream = zfp_stream_open(NULL)

    if tolerance is not None:
        zfp_stream_set_accuracy(stream, tolerance)
    elif precision is not None:
        zfp_stream_set_precision(stream, precision)
    elif rate is not None:
        zfp_stream_set_rate(stream, rate, data_type, len(shape), 0)
    # Try multithreaded
    #if(parallel):
    #    zfp_stream_set_execution(stream, zfp_exec_omp)
    bitstream = stream_open(<void*>&compressed[0], len(compressed))
    zfp_stream_set_bit_stream(stream, bitstream)
    zfp_stream_rewind(stream)
    zfp_decompress(stream, field)
    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bitstream)
    return outdata
