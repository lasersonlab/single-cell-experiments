# Generalize anndata (http://anndata.readthedocs.io/en/latest/) to support Spark RDDs of numpy arrays

import math
import numpy as np
import zarr

import anndata as ad
from anndata.base import BoundRecArr
from zarr_spark import get_chunk_indices, read_zarr_chunk, repartition_chunks


def _read_chunk_csv(csv_file, chunk_size):
    """
    Return a function to read a chunk by coordinates from the given file.
    """

    def read_one_chunk(chunk_index):
        adata = ad.read_csv(csv_file)
        return read_zarr_chunk(adata.X, chunk_size, chunk_index)

    return read_one_chunk


def _read_chunk_zarr(zarr_file, chunk_size):
    """
    Return a function to read a chunk by coordinates from the given file.
    """

    def read_one_chunk(chunk_index):
        adata = ad.read_zarr(zarr_file)
        return read_zarr_chunk(adata.X, chunk_size, chunk_index)

    return read_one_chunk


def _read_chunk_zarr_gcs(gcs_path, chunk_size, gcs_project, gcs_token):
    """
    Return a function to read a chunk by coordinates from the given file.
    """

    def read_one_chunk(chunk_index):
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        adata = ad.read_zarr(store)
        return read_zarr_chunk(adata.X, chunk_size, chunk_index)

    return read_one_chunk


def _write_chunk_zarr(zarr_file):
    """
    Return a function to write a chunk by index to the given file.
    """

    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        index, arr = index_arr
        z = zarr.open(zarr_file, mode="r+")
        x = z["X"]
        chunk_size = x.chunks
        x[chunk_size[0] * index : chunk_size[0] * (index + 1), :] = arr

    return write_one_chunk


def _write_chunk_zarr_gcs(gcs_path, gcs_project, gcs_token):
    """
    Return a function to write a chunk by index to the given file.
    """

    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        index, arr = index_arr
        z = zarr.open(store, mode="r+")
        x = z["X"]
        chunk_size = x.chunks
        x[chunk_size[0] * index : chunk_size[0] * (index + 1), :] = arr

    return write_one_chunk


class AnnDataRdd:
    def __init__(self, sc, adata, rdd, shape, chunks, dtype):
        self.sc = sc
        self.adata = adata
        self.rdd = rdd
        # need to store some metadata about X since adata.X is None so can't retrieve from there
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        # maintain per-partition row counts so we don't have to recompute for repartition_chunks() before saving
        self.partition_row_counts = [chunks[0]] * (shape[0] // chunks[0]) + [
            shape[0] % chunks[0]
        ]

    @classmethod
    def _from_anndata(cls, sc, adata, chunk_size, read_chunk_fn):
        shape = adata.X.shape
        dtype = adata.X.dtype
        ci = get_chunk_indices(shape, chunk_size)
        adata.X = None  # data is stored in the RDD
        chunk_indices = sc.parallelize(ci, len(ci))
        rdd = chunk_indices.map(read_chunk_fn)
        return cls(sc, adata, rdd, shape, chunk_size, dtype)

    @classmethod
    def from_csv(cls, sc, csv_file, chunk_size):
        """
        Read a CSV file as an anndata object (for the metadata) and with the
        data matrix (X) as an RDD of numpy arrays.
        *Note* the anndata object currently also stores the data matrix, which is
        redundant and won't scale. This should be improved, possibly by changing anndata.
        """
        adata = ad.read_csv(csv_file)
        return cls._from_anndata(
            sc, adata, chunk_size, _read_chunk_csv(csv_file, chunk_size)
        )

    @classmethod
    def from_zarr(cls, sc, zarr_file):
        """
        Read a Zarr file as an anndata object (for the metadata) and with the
        data matrix (X) as an RDD of numpy arrays.
        """
        adata = ad.read_zarr(zarr_file)
        chunk_size = zarr.open(zarr_file, mode="r")["X"].chunks
        return cls._from_anndata(
            sc, adata, chunk_size, _read_chunk_zarr(zarr_file, chunk_size)
        )

    @classmethod
    def from_zarr_gcs(cls, sc, gcs_path, gcs_project, gcs_token="cloud"):
        """
        Read a Zarr file from GCS as an anndata object (for the metadata) and with the
        data matrix (X) as an RDD of numpy arrays.
        """
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        adata = ad.read_zarr(store)
        chunk_size = zarr.open(store, mode="r")["X"].chunks
        return cls._from_anndata(
            sc,
            adata,
            chunk_size,
            _read_chunk_zarr_gcs(gcs_path, chunk_size, gcs_project, gcs_token),
        )

    def _write_zarr(self, store, chunks, write_chunk_fn):
        assert chunks[1] == self.adata.n_vars
        # write the metadata out using anndata
        self.adata.write_zarr(store, chunks)
        # write X using Spark
        partitioned_rdd = repartition_chunks(
            self.sc, self.rdd, chunks, self.partition_row_counts
        )  # repartition if needed
        z = zarr.open(store, mode="w")
        shape = (self.adata.n_obs, self.adata.n_vars)
        z.create_dataset("X", shape=shape, chunks=chunks, dtype=self.dtype)

        def index_partitions(index, iterator):
            values = list(iterator)
            assert len(values) == 1  # 1 numpy array per partition
            return [(index, values[0])]

        partitioned_rdd.mapPartitionsWithIndex(index_partitions).foreach(write_chunk_fn)

    def write_zarr(self, zarr_file, chunks):
        """
        Write an anndata object to a Zarr file.
        """
        self._write_zarr(zarr_file, chunks, _write_chunk_zarr(zarr_file))

    def write_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud"):
        """
        Write an anndata object to a Zarr file on GCS.
        """
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        self._write_zarr(
            store, chunks, _write_chunk_zarr_gcs(gcs_path, gcs_project, gcs_token)
        )

    def copy(self):
        return AnnDataRdd(self.adata.copy(), self.rdd)

    def _inplace_subset_var(self, index):
        # similar to same method in AnnData but for the case when X is None
        self.adata._n_vars = np.sum(index)
        self.adata._var = self.adata._var.iloc[index]
        self.adata._varm = BoundRecArr(self.adata._varm[index], self.adata, "varm")
        return None

    def _inplace_subset_obs(self, index, partition_row_counts):
        # similar to same method in AnnData but for the case when X is None
        self.adata._n_obs = np.sum(index)
        self.adata._slice_uns_sparse_matrices_inplace(self.adata._uns, index)
        self.adata._obs = self.adata._obs.iloc[index]
        self.adata._obsm = BoundRecArr(self.adata._obsm[index], self.adata, "obsm")
        self.partition_row_counts = partition_row_counts
        return None
