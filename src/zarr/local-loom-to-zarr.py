import h5py
import math
import numpy as np
import sys
import zarr

# Convert loom (HDF5) to Zarr. Just convert the matrix dataset for the moment.
source = h5py.File("/Downloads/mca.hdf5")
zarr.tree(source)

store = zarr.DirectoryStore('data/mca.zarr')
dest = zarr.group(store=store, overwrite=True)
zarr.copy(source['matrix'], dest, log=sys.stdout)

# Now transpose the Zarr array so it is tall and skinny (loom is short and fat)

def get_chunk_indices(za):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    """
    shape = za.shape
    chunk_size = za.chunks
    return [(i, j) for i in range(int(math.ceil(float(shape[0])/chunk_size[0])))
            for j in range(int(math.ceil(float(shape[1])/chunk_size[1])))]

def read_one_chunk(z, chunk_index):
    """
    Read a zarr chunk specified by coordinates chunk_index=(a,b).
    """
    chunk_size = z.chunks
    return z[chunk_size[0]*chunk_index[0]:chunk_size[0]*(chunk_index[0]+1),chunk_size[1]*chunk_index[1]:chunk_size[1]*(chunk_index[1]+1)]

def write_one_chunk(z, chunk_index, chunk_data):
    """
    Write a zarr chunk specified by coordinates chunk_index=(a,b).
    """
    chunk_size = z.chunks
    z[chunk_size[0]*chunk_index[0]:chunk_size[0]*(chunk_index[0]+1),chunk_size[1]*chunk_index[1]:chunk_size[1]*(chunk_index[1]+1)] = chunk_data

matrix = dest['matrix']
shape = matrix.shape
chunks = matrix.chunks

shape_t = tuple(reversed(matrix.shape))
chunks_t = tuple(reversed(chunks))
z = zarr.open('data/mca.zarr', mode='w', shape=shape_t,
              chunks=chunks_t, dtype=matrix.dtype)

for chunk_index in get_chunk_indices(matrix):
    chunk_data = read_one_chunk(matrix, chunk_index)
    chunk_index_t = tuple(reversed(chunk_index))
    chunk_data_t = np.transpose(chunk_data)
    print("Writing chunk %s to %s" % (chunk_index, chunk_index_t))
    write_one_chunk(z, chunk_index_t, chunk_data_t)
    # if chunk_index[1] == 2:
    #     break

