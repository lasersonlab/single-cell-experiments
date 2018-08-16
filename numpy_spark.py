import builtins
import numbers
import numpy as np
import sys
import zarr

from zarr_spark import get_chunk_indices, read_zarr_chunk, repartition_chunks

from numpy import * # include everything in base numpy

nps = sys.modules[__name__]

def _delegate_to_np(func):
    """Delegate to numpy if the first arg is not an ndarray_rdd"""
    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray_rdd):
            return func(*args, **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)
    return delegated_func

def _delegate_to_np_dist(func):
    """Delegate to numpy if the first arg is not an ndarray_rdd"""
    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray_rdd):
            return _dist_ufunc(func, args, **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)
    return delegated_func

def array_rdd(sc, arr, chunks):
    return ndarray_rdd.from_ndarray(sc, arr, chunks)

def array_rdd_zarr(sc, zarr_file):
    return ndarray_rdd.from_zarr(sc, zarr_file)

def asarray(a):
    if isinstance(a, ndarray_rdd):
        return a.asndarray()
    return np.asarray(a)

# Implement numpy ufuncs
# see https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#available-ufuncs
UFUNC_NAMES = (
    # Math operations (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#math-operations)
    'add',
    'subtract',
    'multiply',
    'divide',
    'logaddexp',
    'logaddexp2',
    'true_divide',
    'floor_divide',
    'negative',
    'positive',
    'power',
    'remainder',
    'mod',
    'fmod',
    #'divmod', # not implemented since returns pair
    'absolute',
    'abs',
    'fabs',
    'rint',
    'sign',
    'heaviside',
    'conj',
    'exp',
    'exp2',
    'log',
    'log2',
    'log10',
    'expm1',
    'log1p',
    'sqrt',
    'square',
    'cbrt',
    'reciprocal',
    # Trigonometric functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#trigonometric-functions)
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'arctan2',
    'hypot',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'deg2rad',
    'rad2deg',
    # Bit-twiddling functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#bit-twiddling-functions)
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'invert',
    'left_shift',
    'right_shift',
    # Comparison functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#comparison-functions)
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'not_equal',
    'equal',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'maximum',
    'minimum',
    'fmax',
    'fmin',
    # Floating functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#floating-functions)
    'isfinite',
    'isinf',
    'isnan',
    'isnat',
    'fabs',
    'signbit',
    'copysign',
    'nextafter',
    'spacing',
    #'modf', # not implemented since returns pair
    'ldexp',
    #'frexp', # not implemented since returns pair
    'fmod',
    'floor',
    'ceil',
    'trunc'
)
for ufunc_name in UFUNC_NAMES:
    ufunc = getattr(np, ufunc_name)
    setattr(nps, ufunc_name, _delegate_to_np_dist(ufunc))

# Implementations of selected functions in the numpy package

@_delegate_to_np
def sum(a, axis=None):
    if axis == 0: # sum of each column
        result = a.rdd.map(lambda x: np.sum(x, axis=0)).collect()
        s = np.sum(result, axis=0)
        rdd = a.rdd.ctx.parallelize([s])
        return a._new(rdd, s.shape, s.shape, partition_row_counts=s.shape)
    elif axis == 1: # sum of each row
        return a._new(a.rdd.map(lambda x: np.sum(x, axis=1)), (a.shape[0],), (a.chunks[0],))
    return NotImplemented

@_delegate_to_np
def mean(a, axis=None):
    if axis == 0: # mean of each column
        result = a.rdd.map(lambda x: (x.shape[0], np.sum(x, axis=0))).collect()
        total_count = builtins.sum([res[0] for res in result])
        mean = np.sum([res[1] for res in result], axis=0) / total_count
        rdd = a.rdd.ctx.parallelize([mean])
        return a._new(rdd, mean.shape, mean.shape, partition_row_counts=mean.shape)
    return NotImplemented

@_delegate_to_np
def median(a):
    # note this is not a distributed implementation
    return np.median(a.asndarray())

# ndarray in Spark

def _read_chunk_from_arr(arr, chunks, chunk_index):
    return arr[chunks[0]*chunk_index[0]:chunks[0]*(chunk_index[0]+1),chunks[1]*chunk_index[1]:chunks[1]*(chunk_index[1]+1)]

def _read_chunk(arr, chunks):
    """
    Return a function to read a chunk by coordinates from the given ndarray.
    """
    def read_one_chunk(chunk_index):
        return _read_chunk_from_arr(arr, chunks, chunk_index)
    return read_one_chunk

def _read_chunk_zarr(zarr_file, chunks):
    """
    Return a function to read a chunk by coordinates from the given file.
    """
    def read_one_chunk(chunk_index):
        z = zarr.open(zarr_file, mode='r')
        return read_zarr_chunk(z, chunks, chunk_index)
    return read_one_chunk

def _write_chunk_zarr(zarr_file):
    """
    Return a function to write a chunk by index to the given file.
    """
    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        index, arr = index_arr
        z = zarr.open(zarr_file, mode='r+')
        chunk_size = z.chunks
        z[chunk_size[0]*index:chunk_size[0]*(index+1),:] = arr
    return write_one_chunk

def _write_chunk_zarr_gcs(gcs_path, gcs_project, gcs_token):
    """
    Return a function to write a chunk by index to the given file.
    """
    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        import gcsfs.mapping
        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        index, arr = index_arr
        z = zarr.open(store, mode='r+')
        chunk_size = z.chunks
        z[chunk_size[0]*index:chunk_size[0]*(index+1),:] = arr
    return write_one_chunk

class ndarray_rdd:
    """A numpy.ndarray backed by a Spark RDD"""

    # Note that this is *not* a subclass of numpy.ndarray. This is because it is unclear whether subclasses
    # must still be memory backed. In particular, when calling np.asarray() for a subclass, it would read the
    # backing memory (and get gibberish, since it was uninitialized) since I couldn't see a way of getting
    # it to use indexing for this call.

    def __init__(self, sc, rdd, shape, chunks, dtype, partition_row_counts=None):
        self.sc = sc
        self.rdd = rdd
        self.ndim = len(shape)
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        if partition_row_counts is None:
            partition_row_counts = [chunks[0]] * (shape[0] // chunks[0])
            remaining = shape[0] % chunks[0]
            if remaining != 0:
                partition_row_counts.append(remaining)
        self.partition_row_counts = partition_row_counts

    def _new(self, rdd, shape=None, chunks=None, dtype=None, partition_row_counts=None):
        if shape is None:
            shape = self.shape
        if chunks is None:
            chunks = self.chunks
        if dtype is None:
            dtype = self.dtype
        if partition_row_counts is None:
            partition_row_counts = self.partition_row_counts
        return ndarray_rdd(self.sc, rdd, shape, chunks, dtype, partition_row_counts)

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, sc, arr, chunks):
        shape = arr.shape
        ci = get_chunk_indices(shape, chunks)
        chunk_indices = sc.parallelize(ci, len(ci))
        rdd = chunk_indices.map(_read_chunk(arr, chunks))
        return cls(sc, rdd, shape, chunks, arr.dtype)

    @classmethod
    def from_zarr(cls, sc, zarr_file):
        """
        Read a Zarr file as an ndarray_rdd object.
        """
        z = zarr.open(zarr_file, mode='r')
        shape, chunks = z.shape, z.chunks
        ci = get_chunk_indices(shape, chunks)
        chunk_indices = sc.parallelize(ci, len(ci))
        rdd = chunk_indices.map(_read_chunk_zarr(zarr_file, chunks))
        return cls(sc, rdd, shape, chunks, z.dtype)

    def asndarray(self):
        local_rows = self.rdd.collect()
        rdd_row_counts = [len(arr) for arr in local_rows]
        assert rdd_row_counts == list(self.partition_row_counts), "RDD row counts: %s; partition row counts: %s" % (rdd_row_counts, self.partition_row_counts)
        arr = np.concatenate(local_rows)
        assert arr.shape[0] == builtins.sum(self.partition_row_counts), "RDD #rows: %s; partition row counts total: %s" % (arr.shape[0], builtins.sum(self.partition_row_counts))
        return arr

    def _write_zarr(self, store, chunks, write_chunk_fn):
        partitioned_rdd = repartition_chunks(self.sc, self.rdd, chunks, self.partition_row_counts) # repartition if needed
        zarr.open(store, mode='w', shape=self.shape, chunks=chunks, dtype=self.dtype)
        def index_partitions(index, iterator):
            values = list(iterator)
            assert len(values) == 1 # 1 numpy array per partition
            return [(index, values[0])]
        partitioned_rdd.mapPartitionsWithIndex(index_partitions).foreach(write_chunk_fn)

    def to_zarr(self, zarr_file, chunks):
        """
        Write an anndata object to a Zarr file.
        """
        self._write_zarr(zarr_file, chunks, _write_chunk_zarr(zarr_file))

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token='cloud'):
        """
        Write an anndata object to a Zarr file on GCS.
        """
        import gcsfs.mapping
        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        self._write_zarr(store, chunks, _write_chunk_zarr_gcs(gcs_path, gcs_project, gcs_token))

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        return nps.mean(self, axis)

    def sum(self, axis=None):
        return nps.sum(self, axis)

    # TODO: more calculation methods here

    # Arithmetic, matrix multiplication, and comparison operations (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations)

    # Python operator overloading, all delegate to ufunc implementations in this package

    # Comparison operators

    def __lt__(self, other):
        return nps.less(self, other, dtype=bool)

    def __le__(self, other):
        return nps.less_equal(self, other, dtype=bool)

    def __gt__(self, other):
        return nps.greater(self, other, dtype=bool)

    def __ge__(self, other):
        return nps.greater_equal(self, other, dtype=bool)

    def __eq__(self, other):
        return nps.equal(self, other, dtype=bool)

    def __ne__(self, other):
        return nps.not_equal(self, other, dtype=bool)

    # Truth value of an array (bool)

    # TODO: __nonzero__

    # Unary operations

    def __neg__(self):
        return nps.negative(self)

    def __pos__(self):
        return nps.positive(self)

    def __abs__(self):
        return nps.abs(self)

    def __invert__(self):
        return nps.invert(self)

    # Arithmetic

    def __add__(self, other):
        return nps.add(self, other)

    def __sub__(self, other):
        return nps.subtract(self, other)

    def __mul__(self, other):
        return nps.multiply(self, other)

    def __div__(self, other):
        return nps.divide(self, other)

    def __truediv__(self, other):
        return nps.true_divide(self, other)

    def __floordiv__(self, other):
        return nps.floor_divide(self, other)

    def __mod__(self, other):
        return nps.mod(self, other)

    # TODO: not implemented since returns pair
    # def __divmod__(self, other):
    #     return nps.div_mod(self, other)

    def __pow__(self, other):
        return nps.power(self, other)

    def __lshift__(self, other):
        return nps.lshift(self, other)

    def __rshift__(self, other):
        return nps.rshift(self, other)

    def __and__(self, other):
        return nps.bitwise_and(self, other)

    def __or__(self, other):
        return nps.bitwise_or(self, other)

    def __xor__(self, other):
        return nps.bitwise_xor(self, other)

    # Arithmetic, in-place

    def __iadd__(self, other):
        return nps.add(self, other, copy=False)

    def __isub__(self, other):
        return nps.subtract(self, other, copy=False)

    def __imul__(self, other):
        return nps.multiply(self, other, copy=False)

    def __idiv__(self, other):
        return nps.multiply(self, other, copy=False)

    def __itruediv__(self, other):
        return nps.true_divide(self, other, copy=False)

    def __ifloordiv__(self, other):
        return nps.floor_divide(self, other, copy=False)

    def __imod__(self, other):
        return nps.mod(self, other, copy=False)

    def __ipow__(self, other):
        return nps.power(self, other, copy=False)

    def __ilshift__(self, other):
        return nps.lshift(self, other, copy=False)

    def __irshift__(self, other):
        return nps.rshift(self, other, copy=False)

    def __iand__(self, other):
        return nps.bitwise_and(self, other, copy=False)

    def __ior__(self, other):
        return nps.bitwise_or(self, other, copy=False)

    def __ixor__(self, other):
        return nps.bitwise_xor(self, other, copy=False)

    # Matrix Multiplication

    # TODO: __matmul__

    # Slicing
    def __getitem__(self, item):
        all_indices = slice(None, None, None)
        if isinstance(item, numbers.Number): # numerical index
            return self.asndarray().__getitem__(item) # TODO: not scalable for large arrays
        elif isinstance(item, np.ndarray) and item.dtype == bool: # boolean index array
            return self.asndarray().__getitem__(item) # TODO: not scalable for large arrays
        elif isinstance(item, ndarray_rdd) and item.dtype == bool: # rdd-backed boolean index array, almost identical to row subset below
            subset = item
            if isinstance(subset, ndarray_rdd): # materialize index RDD to ndarray
                subset = subset.asndarray()
            partition_row_subsets = self._copartition(subset)
            new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
            new_shape = (builtins.sum(new_partition_row_counts),)
            # leave new chunks undefined since they are not necessarily equal-sized
            subset_rdd = self.sc.parallelize(partition_row_subsets, len(partition_row_subsets))
            return self._new(self.rdd.zip(subset_rdd).map(lambda p: p[0][p[1]]), shape=new_shape, partition_row_counts=new_partition_row_counts)
        elif isinstance(item[0], slice) and item[0] == all_indices: # column subset
            if item[1] is np.newaxis: # add new col axis
                new_num_cols = 1
                new_shape = (self.shape[0], new_num_cols)
                new_chunks = (self.chunks[0], new_num_cols)
                return self._new(self.rdd.map(lambda x: x[:, np.newaxis]), shape=new_shape, chunks=new_chunks, partition_row_counts=self.partition_row_counts)
            subset = item[1]
            if isinstance(subset, ndarray_rdd): # materialize index RDD to ndarray
                subset = subset.asndarray()
            new_num_cols = builtins.sum(subset)
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            return self._new(self.rdd.map(lambda x: x[item]), shape=new_shape, chunks=new_chunks, partition_row_counts=self.partition_row_counts)
        elif isinstance(item[1], slice) and item[1] == all_indices: # row subset
            subset = item[0]
            if isinstance(subset, ndarray_rdd): # materialize index RDD to ndarray
                subset = subset.asndarray()
            partition_row_subsets = self._copartition(subset)
            new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
            new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])
            # leave new chunks undefined since they are not necessarily equal-sized
            subset_rdd = self.sc.parallelize(partition_row_subsets, len(partition_row_subsets))
            return self._new(self.rdd.zip(subset_rdd).map(lambda p: p[0][p[1],:]), shape=new_shape, partition_row_counts=new_partition_row_counts)
        return NotImplemented

    def _copartition(self, arr):
        partition_row_subsets = np.split(arr, np.cumsum(self.partition_row_counts)[0:-1])
        if len(partition_row_subsets[-1]) == 0:
            partition_row_subsets = partition_row_subsets[0:-1]
        return partition_row_subsets


def _dist_ufunc(func, args, dtype=None, copy=True):
    a = args[0]
    if len(args) == 1: # unary ufunc
        new_rdd = a.rdd.map(lambda x: func(x))
    elif len(args) == 2: # binary ufunc
        b = args[1]
        if a is b:
            new_rdd = a.rdd.map(lambda x: func(x, x))
        elif isinstance(b, np.ndarray) and a.shape[0] == b.shape[0] and b.shape[1] == 1:
            # args have the same rows but other is an ndarray, so zip to combine
            partition_row_subsets = a._copartition(b)
            repartitioned_other_rdd = a.sc.parallelize(partition_row_subsets, len(partition_row_subsets))
            new_rdd = a.rdd.zip(repartitioned_other_rdd).map(lambda p: func(p[0], p[1]))
        elif isinstance(b, numbers.Number) or isinstance(b, np.ndarray):
            # broadcast case
            new_rdd = a.rdd.map(lambda x: func(x, b))
        elif a.shape[0] == b.shape[0] and a.partition_row_counts == b.partition_row_counts:
            # args have the same rows (and partitioning) so use zip to combine then apply the operator
            new_rdd = a.rdd.zip(b.rdd).map(lambda p: func(p[0], p[1]))
        elif a.shape[0] == b.shape[0] and a.partition_row_counts != b.partition_row_counts and b.shape[1] == 1:
            # args have the same rows but different partitioning, so repartition locally since there's only one column
            partition_row_subsets = a._copartition(b.asndarray())
            repartitioned_other_rdd = a.sc.parallelize(partition_row_subsets, len(partition_row_subsets))
            new_rdd = a.rdd.zip(repartitioned_other_rdd).map(lambda p: func(p[0], p[1]))
        elif b.ndim == 1:
            # materialize 1D RDDs
            return _dist_ufunc(func, (a, b.asndarray()), dtype, copy)
        else:
            print("_dist_ufunc not implemented for %s and %s" % (a, b))
            return NotImplemented
    new_dtype = a.dtype if dtype is None else dtype
    if copy:
        return a._new(new_rdd, dtype=new_dtype)
    else:
        a.rdd = new_rdd
        a.dtype = new_dtype
        return a


