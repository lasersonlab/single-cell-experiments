import pywren

import s3fs.mapping
import zarr

def chunks(file):
    s3 = s3fs.S3FileSystem()
    store = s3fs.mapping.S3Map(file, s3=s3)
    return zarr.open(store, mode="r").chunks

def num_rows(x):
    s3 = s3fs.S3FileSystem()
    store = s3fs.mapping.S3Map('sc-tom-test-data/10x-10k-subset.zarr', s3=s3)
    adata = ad.read_zarr(store)
    return len(adata)

print(chunks('sc-tom-test-data/10x-10k-subset.zarr/X'))

wrenexec = pywren.default_executor()
future = wrenexec.call_async(chunks, 'sc-tom-test-data/10x-10k-subset.zarr/X')
print(future.result())

# Traceback (most recent call last):
#     File "scratch/serverless-zarr-pywren.py", line 28, in <module>
#         print(future.result())
#     File "/Users/tom/workspace/single-cell-experiments/venv/lib/python3.6/site-packages/pywren/future.py", line 202, in result
#         reraise(*self._traceback)
#     File "/Users/tom/workspace/single-cell-experiments/venv/lib/python3.6/site-packages/six.py", line 692, in reraise
#         raise value.with_traceback(tb)
#     File "/var/task/jobrunner.py", line 29, in <module>
#     File "/tmp/pymodules/pywren/serialize/cloudpickle/cloudpickle.py", line 718, in subimport
#     File "/tmp/pymodules/zarr/__init__.py", line 6, in <module>
#     File "/tmp/pymodules/zarr/core.py", line 13, in <module>
#     File "/tmp/pymodules/zarr/util.py", line 13, in <module>
#     ModuleNotFoundError: No module named 'numcodecs'

# To avoid this error, edit `~/.pywren_config` to use a special runtime:

#runtime:
#   s3_bucket: tom-pywren-runtimes
#   s3_key: pywren.runtime/pywren_runtime-3.6-default.meta.json