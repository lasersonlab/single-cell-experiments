# virtualenv venv
# source venv/bin/activate
# pip install zarr
# export PYSPARK_PYTHON=$(pwd)/venv/bin/python
# ~/sw/spark-2.2.1-bin-hadoop2.7/bin/pyspark

import math
import numpy as np
import zarr

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

from zarr_spark import *

# Demo

input_file = '/tmp/mini.zarr'
output_file = '/tmp/svd.zarr'

# Write a small array to a local file
z = zarr.open(input_file, mode='w', shape=(3, 5),
              chunks=(2, 5), dtype='f8', compressor=None)
z[0, :] = [0.0, 1.0, 0.0, 3.0, 0.0]
z[1, :] = [2.0, 0.0, 3.0, 4.0, 5.0]
z[2, :] = [4.0, 0.0, 0.0, 6.0, 7.0]

# Have a look at the data
z[:]

# Use Spark to read and write

spark = SparkSession \
    .builder \
    .appName("zarr-demo") \
    .getOrCreate()
sc = spark.sparkContext

chunk_indices = sc.parallelize(get_chunk_indices(z), 2)

def pr(x):
    print(x)

# Read the entire array. Each chunk is printed by a separate task.
chunk_indices.map(read_chunk(input_file)).foreach(pr)

# Convert the array to a Spark Vector, then run SVD, and convert the result back to an array.
vec = chunk_indices.map(read_chunk(input_file)).flatMap(ndarray_to_vector)
mat = RowMatrix(vec)
svd = mat.computeSVD(2, True)
u = svd.U # U has original number of rows (3) and projected number of cols (2)

# Create a new Zarr file, but only write metadata.
# N.B. no good way currently to find chunk size - have to hardcode it here.
z = zarr.open(output_file, mode='w', shape=(u.numRows(), u.numCols()),
              chunks=(2, 2), dtype='f8', compressor=None)
# Write the entire array. Each chunk is written by a separate task.
u.rows.mapPartitionsWithIndex(vectors_to_ndarray).foreach(write_chunk(output_file))

# Read back locally
z = zarr.open(output_file, mode='r')
z[:]

