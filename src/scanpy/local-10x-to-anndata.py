# python3 -m venv venv3
# source venv3/bin/activate
# pip install numpy zarr
# (cd ~/workspace/anndata; pip install .) # install zarr branch of anndata

import scanpy.api

#adata = scanpy.api.read_10x_h5('/Users/tom/workspace/hdf5-java-cloud/files/1M_neurons_filtered_gene_bc_matrices_h5.h5')
adata = scanpy.api.read_10x_h5('/Downloads/1M_neurons_filtered_gene_bc_matrices_h5.h5')

# Use this to write to HDF5
#adata.write('data/10x.h5ad')

adata.write_zarr('data/10x.zarr', chunks=(5000, 27998))
