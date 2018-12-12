import anndata as ad
import s3fs.mapping
from scanpy.api.pp import log1p
import zappy.executor

executor = zappy.executor.PywrenExecutor(live_viewer=True, exclude_modules=None, ignore_modules=['dash', 'dash_html_components', 'dash_core_components', 'dask', 'google_auth_oauthlib', 'pandas', 'pytest'])

s3 = s3fs.S3FileSystem()
if s3.exists('sc-tom-test-data/10x-log1p.zarr'):
    s3.rm('sc-tom-test-data/10x-log1p.zarr', recursive=True)
input_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_2000/10x.zarr', s3=s3)
input_zarr_X = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_2000/10x.zarr/X', s3=s3)
output_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x-log1p.zarr', s3=s3)

# regular anndata except for X
adata = ad.read_zarr(input_zarr)
adata.X = zappy.executor.from_zarr(executor, input_zarr_X)

log1p(adata) # updates in place

adata.X.to_zarr(output_zarr, adata.X.chunks)
