import s3fs.mapping
import zappy.base as np
import zappy.executor

def get_mask(num_rows, drop_every=10):
    """Return a boolean mask where every n-th entry is False"""
    return np.tile(np.append(np.full((drop_every - 1,), True), False), 1 + num_rows // drop_every)[:num_rows]

s3 = s3fs.S3FileSystem()
if s3.exists('sc-tom-test-data/10x-repartition.zarr'):
    s3.rm('sc-tom-test-data/10x-repartition.zarr', recursive=True)
input_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_2000/10x.zarr/X', s3=s3)
intermediate_zarr = s3fs.mapping.S3Map('sc-tom-test-data/intermediate.zarr', s3=s3)
output_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x-repartition.zarr', s3=s3)

executor = zappy.executor.PywrenExecutor(live_viewer=True, exclude_modules=None, ignore_modules=['dash', 'dash_html_components', 'dash_core_components', 'dask', 'google_auth_oauthlib', 'pandas', 'pytest'])
x = zappy.executor.from_zarr(executor, input_zarr, intermediate_store=intermediate_zarr)

mask = get_mask(x.shape[0])

out = x[mask, :] # subset rows
out.to_zarr(output_zarr, x.chunks)
