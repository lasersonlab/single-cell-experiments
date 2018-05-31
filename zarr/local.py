# export PYSPARK_PYTHON=$(pwd)/venv/bin/python
# ~/sw/spark-2.2.1-bin-hadoop2.7/bin/pyspark

import zarr

import numpy as np

# Write to a local file
z = zarr.open('data/mini.zarr', mode='w', shape=(4, 3),
               chunks=(2, 3), dtype='i4', compressor=None)
z[:, 0] = np.arange(4)
z[:, 1] = np.arange(4, 8)
z[:, 2] = np.arange(8, 12)

# Write to a local file
z = zarr.open('data/mini.zarr', mode='w', shape=(3, 5),
              chunks=(2, 5), dtype='f8', compressor=None)
z[0, :] = [0.0, 1.0, 0.0, 3.0, 0.0]
z[1, :] = [2.0, 0.0, 3.0, 4.0, 5.0]
z[2, :] = [4.0, 0.0, 0.0, 6.0, 7.0]

# Read in parallel
# To keep things simple each task reads a single chunk

import zarr
import numpy as np

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .appName("zarr-demo") \
    .getOrCreate()
sc = spark.sparkContext

chunks = sc.parallelize(((0, 0), (1, 0)), 2)

input_file = '/Users/tom/workspace/hdf5-experiments/zarr/data/mini.zarr'
output_file = '/Users/tom/workspace/hdf5-experiments/zarr/data/svd.zarr'

def noop(x):
    pass

def pr(x):
    print(x, type(x))

def read_chunk(x):
    """
    Read a zarr chunk specified by coordinates x=(a,b) (2D-only).
    """
    z = zarr.open(input_file, mode='r')
    chunk_size = z.chunks
    return z[chunk_size[0]*x[0]:chunk_size[0]*(x[0]+1),chunk_size[1]*x[1]:chunk_size[1]*(x[1]+1)]

chunks.map(read_chunk).foreach(pr)

def ndarray_to_vector(arr):
    """
    Convert a numpy array to a Spark Vector
    """
    return map(lambda row : Vectors.dense(row), arr.tolist())

def vectors_to_ndarray(index, vs):
    """
    Convert a partition index and list of Spark Vectors to a pair of index and numpy array.
    """
    return [(index, np.array([v.toArray() for v in vs]))]

vec = chunks.map(read_chunk).flatMap(ndarray_to_vector)
mat = RowMatrix(vec)
svd = mat.computeSVD(2, True)

u = svd.U # U has original number of rows (3) and projected number of cols (2)
u.rows.mapPartitions(vectors_to_ndarray).foreach(pr)

# Create a new Zarr file, but only write metadata
z = zarr.open(output_file, mode='w', shape=(3, 2),
              chunks=(2, 2), dtype='f8', compressor=None)

# Write each partition in a task
def write_chunk(x):
    """
    Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
    overlap other chunks
    """
    index, arr = x
    z = zarr.open(output_file, mode='r+')
    chunk_size = z.chunks
    z[chunk_size[0]*index:chunk_size[0]*(index+1),:] = arr

u.rows.mapPartitionsWithIndex(vectors_to_ndarray).foreach(write_chunk)

# Read back locally
z = zarr.open(output_file, mode='r')
z[:]