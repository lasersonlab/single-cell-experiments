# Run in Jupyter on a dask cluster (see scripts/start-dask-cluster)

import anndata as ad
import dask.array as da
import numpy as np

from dask.distributed import Client # pip install dask distributed
import gcsfs.mapping

def colsum(zarr_file):
    print("Running colsum for %s" % zarr_file)
    client = Client()
    gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
    store = gcsfs.mapping.GCSMap(zarr_file, gcs=gcs)
    adata = ad.read_zarr(store)
    adata.X = da.from_zarr(store, component='X')

    s = np.sum(adata.X, axis=0)
    s.compute()

# Try different chunk sizes to see if any are significantly faster (all within about 10% of each other; time dominated by zarr)
%time colsum('ll-sc-data-bkup/10x/anndata_zarr_1000/10x.zarr')
%time colsum('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr')
%time colsum('ll-sc-data-bkup/10x/anndata_zarr_4000/10x.zarr')
%time colsum('ll-sc-data-bkup/10x/anndata_zarr/10x.zarr')

