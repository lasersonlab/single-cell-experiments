# Generalize anndata (http://anndata.readthedocs.io/en/latest/) to support Spark RDDs of numpy arrays

import anndata as ad
import math

def get_chunk_indices(shape, chunk_size):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    Note that unlike Zarr the chunk size must be explicitly set.
    """
    return [(i, j) for i in range(int(math.ceil(float(shape[0])/chunk_size[0])))
            for j in range(int(math.ceil(float(shape[1])/chunk_size[1])))]

def read_chunk(csv_file, chunk_size):
    """
    Return a function to read a chunk by coordinates from the given file.
    """
    def read_one_chunk(chunk_index):
        # TODO: load from Zarr so only the relevant chunk is loaded
        adata = ad.read_csv(csv_file)
        return adata.X[chunk_size[0]*chunk_index[0]:chunk_size[0]*(chunk_index[0]+1),chunk_size[1]*chunk_index[1]:chunk_size[1]*(chunk_index[1]+1)]
    return read_one_chunk


class AnnDataRdd:
    def __init__(self, adata, rdd):
        self.adata = adata
        self.rdd = rdd

    # TODO: load from Zarr, not CSV
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
        chunk_indices = sc.parallelize(ci, len(ci))
        rdd = chunk_indices.map(read_chunk(csv_file, chunk_size))
        return cls(adata, rdd)

    def copy(self):
        return AnnDataRdd(self.adata.copy(), self.rdd)
