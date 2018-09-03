import pywren
import s3fs

def ls_bucket(bucket):
    s3 = s3fs.S3FileSystem()
    return s3.ls(bucket)

print(ls_bucket('sc-tom-test-data'))

wrenexec = pywren.default_executor()
future = wrenexec.call_async(ls_bucket, 'sc-tom-test-data')
print(future.result())