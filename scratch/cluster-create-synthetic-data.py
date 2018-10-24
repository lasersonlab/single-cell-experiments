import gcsfs.mapping
import numpy as np
import pyspark
import zappy.spark
import zarr

sc = pyspark.SparkContext()

gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
input = 'll-sc-data-bkup/10x/anndata_zarr/10x_10000000.zarr'
output = 'll-sc-data-bkup/10x/anndata_zarr/10x_100000000.zarr'
ncopies = 10

input_file = gcsfs.mapping.GCSMap(input, gcs=gcs)
output_file = gcsfs.mapping.GCSMap(output, gcs=gcs)

# var: copy unchanged
input_file_var = gcsfs.mapping.GCSMap('{}/var'.format(input), gcs=gcs)
output_file_var = gcsfs.mapping.GCSMap('{}/var'.format(output), gcs=gcs)

zarr.copy_store(input_file_var, output_file_var)

# obs: create n copies locally since it is small enough (and it is not chunked evenly)
# input_file_obs = gcsfs.mapping.GCSMap('{}/obs'.format(input), gcs=gcs)
# output_file_obs = gcsfs.mapping.GCSMap('{}/obs'.format(output), gcs=gcs)
#
# obs = zarr.open(input_file_obs, mode='r')
# obs_ncopies = np.hstack((obs[:], ) * ncopies)
#
# out_root = zarr.group(store=output_file)
# obs_out = out_root.empty('obs', shape=obs_ncopies.shape, chunks=obs.chunks, dtype=obs.dtype)
# obs_out[:] = obs_ncopies

# X: use zappy to create n copies in parallel
input_file_X = gcsfs.mapping.GCSMap('{}/X'.format(input), gcs=gcs)
output_file_X = gcsfs.mapping.GCSMap('{}/X'.format(output), gcs=gcs)

X = zappy.spark.from_zarr(sc, input_file_X)
X.to_zarr(output_file_X, X.chunks, ncopies=ncopies)

