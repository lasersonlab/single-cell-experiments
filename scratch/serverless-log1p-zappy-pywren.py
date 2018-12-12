import s3fs.mapping
import zappy.base as np
import zappy.executor

s3 = s3fs.S3FileSystem()
if s3.exists('sc-tom-test-data/10x-log1p.zarr'):
    s3.rm('sc-tom-test-data/10x-log1p.zarr', recursive=True)
input_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_2000/10x.zarr/X', s3=s3)
output_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x-log1p.zarr', s3=s3)

executor = zappy.executor.PywrenExecutor(live_viewer=True, exclude_modules=None, ignore_modules=['dash', 'dash_html_components', 'dash_core_components', 'dask', 'google_auth_oauthlib', 'pandas', 'pytest'])
x = zappy.executor.from_zarr(executor, input_zarr)

out = np.log1p(x)
out.to_zarr(output_zarr, x.chunks)
