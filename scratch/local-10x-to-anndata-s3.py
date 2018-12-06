# # On an EC2 m5a.4xlarge instance (16 CPUs, 64GB memory) with Ubuntu 18.04
# sudo apt-get update
# sudo apt-get install -y python3-pip python3-tk
# pip3 install s3fs numpy scanpy zarr

# Copy 10x data locally
# wget https://storage.googleapis.com/ll-sc-data-bkup/10x/1M_neurons_filtered_gene_bc_matrices_h5.h5

# python3

import s3fs
import scanpy.api

s3 = s3fs.S3FileSystem()
s3.ls('sc-tom-test-data')
#store = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_1000/10x.zarr', s3=s3)
#store = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_2000/10x.zarr', s3=s3)
#store = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr_4000/10x.zarr', s3=s3)
store = s3fs.mapping.S3Map('sc-tom-test-data/10x/anndata_zarr/10x.zarr', s3=s3)

adata = scanpy.api.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')
#adata = scanpy.api.read_10x_h5('/Downloads/1M_neurons_filtered_gene_bc_matrices_h5.h5')

# Use this to write to HDF5
#adata.write('data/10x.h5ad')

# Use this for a subset
#adata_subset = adata[:10000, ]

#adata.write_zarr(store, chunks=(1000, 27998))
#adata.write_zarr(store, chunks=(2000, 27998))
#adata.write_zarr(store, chunks=(4000, 27998))
adata.write_zarr(store, chunks=(10000, 27998))
