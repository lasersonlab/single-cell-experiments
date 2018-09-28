import anndata as ad
import pywren
import s3fs.mapping
from scanpy.api.pp import log1p
import zarr
import zap.base as np
import zap.executor.array

executor = zap.executor.array.PywrenExecutor(exclude_modules=["gcsfs", "dask", "pytest", "oauthlib", "requests_oauthlib", "google_auth_oauthlib", "pytest"])

s3 = s3fs.S3FileSystem()
input_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x.zarr', s3=s3)
input_zarr_X = s3fs.mapping.S3Map('sc-tom-test-data/10x.zarr/X', s3=s3)
output_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x-log1p.zarr', s3=s3)

# regular anndata except for X
adata = ad.read_zarr(input_zarr)
adata.X = zap.executor.array.from_zarr(executor, input_zarr_X)

log1p(adata) # updates in place

adata.X.to_zarr(output_zarr, adata.X.chunks)
