# export PYSPARK_PYTHON=$(pwd)/venv/bin/python
# ~/sw/spark-2.2.1-bin-hadoop2.7/bin/pyspark --master local[2]

import math
import numpy as np
import zarr

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import *

from zarr_spark import *

# Use Spark to read and write

spark = SparkSession \
    .builder \
    .appName("zarr-analytics") \
    .getOrCreate()
sc = spark.sparkContext

input_file = '/Users/tom/workspace/hdf5-java-cloud/files/mcat.zarr'
output_file = '/tmp/mcatnorm.zarr'

z = zarr.open(input_file)
ci = get_chunk_indices(z)[0:2] # during dev
chunk_indices = sc.parallelize(ci, len(ci))
zarr_rdd = chunk_indices.map(read_chunk(input_file))

# load as Spark Vectors
vec = zarr_rdd.flatMap(ndarray_to_vector)

# normalize
def log_normalize(vec, scaleFactor=1e4):
    s = vec.toArray().sum()
    return Vectors.dense([math.log1p(v / s * scaleFactor) for v in vec])

vec_norm = vec.map(log_normalize)

# save as Zarr
z = zarr.open(output_file, mode='w', shape=z.shape,
              chunks=z.chunks, dtype=z.dtype)
save_as_zarr_file(vec.mapPartitionsWithIndex(vectors_to_ndarray), output_file)

