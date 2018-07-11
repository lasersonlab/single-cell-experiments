# Run in Jupyter on a dask cluster (see scripts/start-dask-cluster)

import anndata as ad
import dask.array as da

from dask.distributed import Client # pip install dask distributed
import gcsfs.mapping

from scanpy.api.pp import recipe_zheng17

client = Client()

# regular anndata except for X
gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr', gcs=gcs)
adata = ad.read_zarr(store)
adata.X = da.from_zarr(store, component='X')

%%time
recipe_zheng17(adata)
store_out = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr/10x-recipe-dask.zarr', gcs=gcs)
adata.X.to_zarr(store_out, overwrite=True)
