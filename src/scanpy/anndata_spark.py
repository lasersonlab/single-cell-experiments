# Generalize anndata (http://anndata.readthedocs.io/en/latest/) to support Spark RDDs of numpy arrays

import anndata as ad
import math
import numpy as np
import zarr

from anndata.base import BoundRecArr

def get_chunk_indices(shape, chunk_size):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    Note that unlike Zarr the chunk size must be explicitly set.
    """
    return [(i, j) for i in range(int(math.ceil(float(shape[0])/chunk_size[0])))
            for j in range(int(math.ceil(float(shape[1])/chunk_size[1])))]

def read_chunk_csv(csv_file, chunk_size):
    """
    Return a function to read a chunk by coordinates from the given file.
    """
    def read_one_chunk(chunk_index):
        adata = ad.read_csv(csv_file)
        return adata.X[chunk_size[0]*chunk_index[0]:chunk_size[0]*(chunk_index[0]+1),chunk_size[1]*chunk_index[1]:chunk_size[1]*(chunk_index[1]+1)]
    return read_one_chunk

def read_chunk_zarr(zarr_file, chunk_size):
    """
    Return a function to read a chunk by coordinates from the given file.
    """
    def read_one_chunk(chunk_index):
        adata = ad.read_zarr(zarr_file)
        return adata.X[chunk_size[0]*chunk_index[0]:chunk_size[0]*(chunk_index[0]+1),chunk_size[1]*chunk_index[1]:chunk_size[1]*(chunk_index[1]+1)]
    return read_one_chunk


class AnnDataRdd:
    def __init__(self, adata, rdd):
        self.adata = adata
        self.rdd = rdd

    @classmethod
    def from_csv(cls, sc, csv_file, chunk_size):
        """
        Read a CSV file as an anndata object (for the metadata) and with the
        data matrix (X) as an RDD of numpy arrays.
        *Note* the anndata object currently also stores the data matrix, which is
        redundant and won't scale. This should be improved, possibly by changing anndata.
        """
        adata = ad.read_csv(csv_file)
        ci = get_chunk_indices(adata.X.shape, chunk_size)
        adata.X = None # data is stored in the RDD
        chunk_indices = sc.parallelize(ci, len(ci))
        rdd = chunk_indices.map(read_chunk_csv(csv_file, chunk_size))
        return cls(adata, rdd)

    @classmethod
    def from_zarr(cls, sc, zarr_file):
        adata = ad.read_zarr(zarr_file)
        chunk_size = zarr.open(zarr_file, mode='r')['X'].chunks
        ci = get_chunk_indices(adata.X.shape, chunk_size)
        adata.X = None # data is stored in the RDD
        chunk_indices = sc.parallelize(ci, len(ci))
        rdd = chunk_indices.map(read_chunk_zarr(zarr_file, chunk_size))
        return cls(adata, rdd)

    def copy(self):
        return AnnDataRdd(self.adata.copy(), self.rdd)

    def _inplace_subset_var(self, index):
        # similar to same method in AnnData but for the case when X is None
        self.adata._n_vars = np.sum(index)
        self.adata._var = self.adata._var.iloc[index]
        self.adata._varm = BoundRecArr(self.adata._varm[index], self.adata, 'varm')
        return None

    def _inplace_subset_obs(self, index):
        # similar to same method in AnnData but for the case when X is None
        self.adata._n_obs = np.sum(index)
        self.adata._slice_uns_sparse_matrices_inplace(self.adata._uns, index)
        self.adata._obs = self.adata._obs.iloc[index]
        self.adata._obsm = BoundRecArr(self.adata._obsm[index], self.adata, 'obsm')
        return None
