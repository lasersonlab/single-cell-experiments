# Run on a GCE instance
# n1-ultramem-40 - 961GB mem, 40GB disk
# gcloud beta compute --project=hca-scale instances create instance-1 --zone=us-east1-b --machine-type=n1-ultramem-40 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=218219996328-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --min-cpu-platform=Intel\ Broadwell --image=debian-9-stretch-v20181011 --image-project=debian-cloud --boot-disk-size=40GB --boot-disk-type=pd-ssd --boot-disk-device-name=instance-1
#
# sudo apt-get update && sudo apt-get install -y git python3-pip python3-tk
# pip3 install scanpy==1.2.2
# pip3 uninstall -y anndata
# pip3 install git+https://github.com/tomwhite/anndata@zarr
# pip3 install gcsfs
# pip3 list

import anndata as ad
import gcsfs.mapping

from scanpy.api.pp import log1p
from scanpy.api.pp import recipe_zheng17

gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_2000/10x.zarr', gcs=gcs)
#store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/10x-10k-subset.zarr', gcs=gcs)
output = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr_out/10x.zarr', gcs=gcs)
adata = ad.read_zarr(store)

import time
start = time.time()

adata.X = adata.X[:] # materialize in memory since Zarr doesn't support all the operations scanpy calls

recipe_zheng17(adata)

adata.write_zarr(output, chunks=(2000, adata.n_vars))

end = time.time()
print(end - start)

# 1080.5862543582916
# This is 18 minutes - note that it *doesn't* write back to cloud storage