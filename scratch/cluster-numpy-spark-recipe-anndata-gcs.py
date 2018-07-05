import pyspark

import anndata as ad
import gcsfs.mapping
import numpy_spark as np # numpy_spark includes everything in numpy, with some overrides and new functions

from scanpy.api.pp import recipe_zheng17

sc = pyspark.SparkContext()

# regular anndata except for X
gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr', gcs=gcs)
storeX = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr/X', gcs=gcs)
adata = ad.read_zarr(store)
adata.X = np.array_rdd_zarr(sc, storeX)

recipe_zheng17(adata)

adata.X.to_zarr_gcs('ll-sc-data-bkup/10x/anndata_zarr/10x-recipe.zarr', (2000, adata.n_vars), 'hca-scale')
