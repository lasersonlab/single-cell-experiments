import pyspark

import anndata as ad
import gcsfs.mapping
import numpy_spark as np # numpy_spark includes everything in numpy, with some overrides and new functions

from scanpy.api.pp import log1p

sc = pyspark.SparkContext()

# regular anndata except for X
gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr', gcs=gcs)
storeX = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr/X', gcs=gcs)
adata = ad.read_zarr(store)
adata.X = np.array_rdd_zarr(sc, storeX)

log1p(adata) # updates in place

adata.X.to_zarr_gcs('ll-sc-data-bkup/10x/anndata_zarr/10x-log1p.zarr', adata.X.chunks, 'hca-scale')
