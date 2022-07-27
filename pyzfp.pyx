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

  cdef uint ZFP_HEADER_NONE 
  cdef uint ZFP_HEADER_MAGIC
  cdef uint ZFP_HEADER_META 
  cdef uint ZFP_HEADER_MODE
  cdef uint ZFP_HEADER_FULL 

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

  cdef size_t zfp_write_header(
      zfp_stream* stream, #/* compressed stream */
      const zfp_field* field,    #/* field metadata */
      uint umask
  );

  cdef size_t zfp_read_header(
      zfp_stream* stream, #/* compressed stream */
      const zfp_field* field,    #/* field metadata */
      uint umask
  );

cdef void* raw_pointer_double(arr) except NULL:
    assert(arr.dtype==np.float64)
    assert(arr.flags.forc) # if this isn't true, ravel will make a copy
    cdef double[::1] mview = arr.ravel(order='A')
    return <void*>&mview[0]

cdef void* raw_pointer_float(arr) except NULL:
    assert(arr.flags.forc) # if this isn't true, ravel will make a copy
    assert(arr.dtype == np.float32)
    cdef float[::1] mview = arr.ravel(order='A')
    return <void*>&mview[0]

cdef void* raw_pointer(arr):
    if arr.dtype == np.float32:
        return raw_pointer_float(arr)
    else:
        return raw_pointer_double(arr)

zfp_types = {
  np.dtype('float32'): zfp_type_float, 
  np.dtype('float64'): zfp_type_double,
  zfp_type_float: np.dtype('float32'), 
  zfp_type_double: np.dtype('float64'),
}

class EncodingError(Exception):
  pass

class DecodingError(Exception):
  pass

cdef zfp_field* init_field(np.ndarray indata):
    data_type = zfp_types[indata.dtype]
    numdims = len((<object> indata).shape)

    # Had to do this awkward construction
    # because list(indata.shape) had an error.
    shape = []
    for i in range(indata.ndim):
      shape.append(indata.shape[i])

    if indata.flags.c_contiguous:
      shape = shape[::-1]

    if numdims == 1:
        return zfp_field_1d(raw_pointer(indata), data_type, shape[0])
    elif numdims == 2:
        return zfp_field_2d(raw_pointer(indata), data_type, shape[0], shape[1])
    elif numdims == 3:
        return zfp_field_3d(raw_pointer(indata), data_type, shape[0], shape[1], shape[2])
    elif numdims == 4:
       return zfp_field_4d(raw_pointer(indata), data_type, shape[0], shape[1], shape[2], shape[3])
    else:
        raise ValueError("Invalid number of dimensions (valid: 1-4)")

def compress(indata, tolerance=None, precision=None, rate=None, parallel=True):
    """
    Compress a numpy array using zfp.

    Parameters
    ----------
    indata : numpy.ndarray
        The data to compress.
    tolerance : float, optional
        The tolerance for the compressed data.
        This will use ZFP in fixed-accuracy mode.
        https://zfp.readthedocs.io/en/latest/modes.html#mode-fixed-accuracy
        One of tolerance, precision, or rate must be specified.
    precision : int, optional
        The precision of the compressed data.
        This will use ZFP in fixed-precision mode.
        https://zfp.readthedocs.io/en/latest/modes.html#fixed-precision-mode
        One of tolerance, precision, or rate must be specified.
    rate : float, optional
        The rate of the compressed data.
        This will use ZFP in fixed-rate mode.
        https://zfp.readthedocs.io/en/latest/modes.html#fixed-rate-mode
        One of tolerance, precision, or rate must be specified.
    parallel : bool, optional, default True
        Whether to use parallel compression.
        This will use ZFP in parallel mode.
    
    Returns
    -------
    outdata : Cython.memoryview
        The compressed data.
    """
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
    cdef size_t header_bytes = zfp_write_header(stream, field, ZFP_HEADER_FULL)

    if header_bytes == 0:
      raise EncodingError("Unable to write header.")

    cdef size_t zfpsize = zfp_compress(stream, field)

    if zfpsize == 0:
      raise EncodingError("Unable to write byte stream.")

    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bitstream)
    return buff[:header_bytes + zfpsize]

def decompress(const unsigned char[::1] compressed, parallel=True, order='C'):
    """
    Decompress a numpy array using zfp.

    Parameters
    ----------
    compressed : Cython.memoryview
        The compressed data.
    parallel : bool, optional, default False
        Whether to use parallel decompression.
        This will use ZFP in parallel mode.
    order : str, optional, default 'C'
        The order of the decompressed data.
        Must be either C or F(ortran).
    
    Returns
    -------
    outdata : numpy.ndarray
        The decompressed data.
    """

    order = order.upper()
    assert(order in ('C', 'F'))
    
    stream = zfp_stream_open(NULL)
    bitstream = stream_open(<void*>&compressed[0], len(compressed))
    zfp_stream_set_bit_stream(stream, bitstream)
    zfp_stream_rewind(stream)

    cdef zfp_field header_field
    cdef size_t header_bytes = zfp_read_header(stream, &header_field, ZFP_HEADER_FULL)

    if header_bytes == 0:
        raise DecodingError("Unable to read stream header.")

    shape = [ header_field.nx, header_field.ny, header_field.nz, header_field.nw ]
    shape = [ dim for dim in shape if dim > 0 ]

    if order == "C":
        shape = shape[::-1]

    outdata = np.zeros(shape, dtype=zfp_types[header_field.type], order=order)
    field = init_field(outdata)
    if(parallel):
        try:
            zfp_stream_set_execution(stream, zfp_exec_omp)
        except:
            raise ValueError("Parallel decompression not supported on this platform")

    zfp_decompress(stream, field)
    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bitstream)
    return outdata
