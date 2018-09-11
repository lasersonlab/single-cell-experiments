# python3 -m venv venv3
# source venv3/bin/activate
# pip install s3fs scanpy==1.2.2 zarr==2.2.0 git+https://github.com/tomwhite/anndata@zarr

import s3fs
import scanpy.api

s3 = s3fs.S3FileSystem()
s3.ls('sc-tom-test-data')
store = s3fs.mapping.S3Map('sc-tom-test-data/10x.zarr', s3=s3)

adata = scanpy.api.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')
#adata = scanpy.api.read_10x_h5('/Downloads/1M_neurons_filtered_gene_bc_matrices_h5.h5')

# Use this to write to HDF5
#adata.write('data/10x.h5ad')

# Use this for a subset
#adata_subset = adata[:10000, ]

adata.write_zarr(store, chunks=(10000, 27998))
