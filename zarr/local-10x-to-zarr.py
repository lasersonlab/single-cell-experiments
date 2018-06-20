import h5py
import sys
import zarr

# Convert 10x (HDF5) to Zarr
source = h5py.File("/Downloads/1M_neurons_filtered_gene_bc_matrices_h5.h5")
zarr.tree(source)

store = zarr.DirectoryStore('data/10x.zarr')
dest = zarr.group(store=store, overwrite=True)
# following fails if without_attrs=False (the default), possibly related to https://github.com/h5py/h5py/issues/973
zarr.copy_all(source, dest, log=sys.stdout, without_attrs=True)


