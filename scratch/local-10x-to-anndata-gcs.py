# # On a GCP ubuntu node n1-standard-16 (16 vCPUs, 60 GB memory)
# sudo apt-get install -y python3-pip python3-tk git
# pip3 install gcsfs numpy scanpy zarr
# (git clone https://github.com/tomwhite/anndata; cd anndata; git checkout -b zarr origin/zarr; pip3 install .) # install zarr branch of anndata

# Copy 10x data locally
# gsutil cp gs://ll-sc-data/10x/1M_neurons_filtered_gene_bc_matrices_h5.h5 1M_neurons_filtered_gene_bc_matrices_h5.h5

# python3

import gcsfs.mapping
import scanpy.api

gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
store = gcsfs.mapping.GCSMap('ll-sc-data/10x/anndata_zarr/10x.zarr', gcs=gcs)

adata = scanpy.api.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')
adata.write_zarr(store, chunks=(10000, 27998))