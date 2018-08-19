import numpy as np
import sys

from numpy import *  # include everything in base numpy

npd = sys.modules[__name__]


def _delegate_to_np(func):
    """Delegate to numpy if the first arg is not an ndarray_dist"""

    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray_dist):
            return func(*args, **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)

    return delegated_func


def _delegate_to_np_dist(func):
    """Delegate to numpy if the first arg is not an ndarray_dist"""

    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray_dist):
            return args[0]._dist_ufunc(func, args, **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)

    return delegated_func


# Implement numpy ufuncs
# see https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#available-ufuncs
UFUNC_NAMES = (
    # Math operations (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#math-operations)
    "add",
    "subtract",
    "multiply",
    "divide",
    "logaddexp",
    "logaddexp2",
    "true_divide",
    "floor_divide",
    "negative",
    "positive",
    "power",
    "remainder",
    "mod",
    "fmod",
    # 'divmod', # not implemented since returns pair
    "absolute",
    "abs",
    "fabs",
    "rint",
    "sign",
    "heaviside",
    "conj",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    # Trigonometric functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#trigonometric-functions)
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    # Bit-twiddling functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#bit-twiddling-functions)
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
    # Comparison functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#comparison-functions)
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    # Floating functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#floating-functions)
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "fabs",
    "signbit",
    "copysign",
    "nextafter",
    "spacing",
    # 'modf', # not implemented since returns pair
    "ldexp",
    # 'frexp', # not implemented since returns pair
    "fmod",
    "floor",
    "ceil",
    "trunc",
)
for ufunc_name in UFUNC_NAMES:
    ufunc = getattr(np, ufunc_name)
    setattr(npd, ufunc_name, _delegate_to_np_dist(ufunc))

# Implementations of selected functions in the numpy package


@_delegate_to_np
def sum(a, axis=None):
    return a.sum(axis)


@_delegate_to_np
def mean(a, axis=None):
    return a.mean(axis)


@_delegate_to_np
def median(a):
    # note this is not a distributed implementation
    return np.median(a.asndarray())


class ndarray_dist:

    # Load and store methods

    @classmethod
    def from_ndarray(cls, sc, arr, chunks):
        return NotImplemented

    @classmethod
    def from_zarr(cls, sc, zarr_file):
        return NotImplemented

    def asndarray(self):
        return NotImplemented

    def to_zarr(self, zarr_file, chunks):
        """
        Write an ndarray_dist object to a Zarr file.
        """
        return NotImplemented

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud"):
        """
        Write an ndarray_dist object to a Zarr file on GCS.
        """
        return NotImplemented

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        return NotImplemented

    def sum(self, axis=None):
        return NotImplemented

    # TODO: more calculation methods here

    # Distributed ufunc internal implementation

    @classmethod
    def _dist_ufunc(cls, func, args, dtype=None, copy=True):
        return NotImplemented

    # Slicing implementation

    def __getitem__(self, item):
        return NotImplemented

    # Arithmetic, matrix multiplication, and comparison operations (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations)

    # Python operator overloading, all delegate to ufunc implementations in this package

    # Comparison operators

    def __lt__(self, other):
        return npd.less(self, other, dtype=bool)

    def __le__(self, other):
        return npd.less_equal(self, other, dtype=bool)

    def __gt__(self, other):
        return npd.greater(self, other, dtype=bool)

    def __ge__(self, other):
        return npd.greater_equal(self, other, dtype=bool)

    def __eq__(self, other):
        return npd.equal(self, other, dtype=bool)

    def __ne__(self, other):
        return npd.not_equal(self, other, dtype=bool)

    # Truth value of an array (bool)

    # TODO: __nonzero__

    # Unary operations

    def __neg__(self):
        return npd.negative(self)

    def __pos__(self):
        return npd.positive(self)

    def __abs__(self):
        return npd.abs(self)

    def __invert__(self):
        return npd.invert(self)

    # Arithmetic

    def __add__(self, other):
        return npd.add(self, other)

    def __sub__(self, other):
        return npd.subtract(self, other)

    def __mul__(self, other):
        return npd.multiply(self, other)

    def __div__(self, other):
        return npd.divide(self, other)

    def __truediv__(self, other):
        return npd.true_divide(self, other)

    def __floordiv__(self, other):
        return npd.floor_divide(self, other)

    def __mod__(self, other):
        return npd.mod(self, other)

    # TODO: not implemented since returns pair
    # def __divmod__(self, other):
    #     return npd.div_mod(self, other)

    def __pow__(self, other):
        return npd.power(self, other)

    def __lshift__(self, other):
        return npd.lshift(self, other)

    def __rshift__(self, other):
        return npd.rshift(self, other)

    def __and__(self, other):
        return npd.bitwise_and(self, other)

    def __or__(self, other):
        return npd.bitwise_or(self, other)

    def __xor__(self, other):
        return npd.bitwise_xor(self, other)

    # Arithmetic, in-place

    def __iadd__(self, other):
        return npd.add(self, other, copy=False)

    def __isub__(self, other):
        return npd.subtract(self, other, copy=False)

    def __imul__(self, other):
        return npd.multiply(self, other, copy=False)

    def __idiv__(self, other):
        return npd.multiply(self, other, copy=False)

    def __itruediv__(self, other):
        return npd.true_divide(self, other, copy=False)

    def __ifloordiv__(self, other):
        return npd.floor_divide(self, other, copy=False)

    def __imod__(self, other):
        return npd.mod(self, other, copy=False)

    def __ipow__(self, other):
        return npd.power(self, other, copy=False)

    def __ilshift__(self, other):
        return npd.lshift(self, other, copy=False)

    def __irshift__(self, other):
        return npd.rshift(self, other, copy=False)

    def __iand__(self, other):
        return npd.bitwise_and(self, other, copy=False)

    def __ior__(self, other):
        return npd.bitwise_or(self, other, copy=False)

    def __ixor__(self, other):
        return npd.bitwise_xor(self, other, copy=False)

    # Matrix Multiplication

    # TODO: __matmul__
