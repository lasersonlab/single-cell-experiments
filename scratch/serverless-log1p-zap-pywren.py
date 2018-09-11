import pywren
import s3fs.mapping
import zarr
import zap.base as np
import zap.executor.array

s3 = s3fs.S3FileSystem()
input_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x.zarr/X', s3=s3)
output_zarr = s3fs.mapping.S3Map('sc-tom-test-data/10x-log1p.zarr', s3=s3)

executor = zap.executor.array.PywrenExecutorWrapper(pywren.default_executor())
x = zap.executor.array.ndarray_executor.from_zarr(executor, input_zarr)

out = np.log1p(x)
out.to_zarr(output_zarr, x.chunks)
