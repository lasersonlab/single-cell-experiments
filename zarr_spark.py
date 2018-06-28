import math
import numpy as np
import zarr

from pyspark.mllib.linalg import Vectors

# Utility functions for reading and writing a Zarr array chunk; these are designed to be run as Spark tasks.
# Currently it is assumed that all arrays are 2D and that rows are not chunked. Also each task reads/writes a single chunk.

def get_chunk_indices(za):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    """
    shape = za.shape
    chunk_size = za.chunks
    return [(i, j) for i in range(int(math.ceil(float(shape[0])/chunk_size[0])))
            for j in range(int(math.ceil(float(shape[1])/chunk_size[1])))]

def read_chunk(file):
    """
    Return a function to read a chunk by coordinates from the given file.
    """
    def read_one_chunk(chunk_index):
        """
        Read a zarr chunk specified by coordinates chunk_index=(a,b).
        """
        z = zarr.open(file, mode='r')
        chunk_size = z.chunks
        return z[chunk_size[0]*chunk_index[0]:chunk_size[0]*(chunk_index[0]+1),chunk_size[1]*chunk_index[1]:chunk_size[1]*(chunk_index[1]+1)]
    return read_one_chunk

def write_chunk(file):
    """
    Return a function to write a chunk by index to the given file.
    """
    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        index, arr = index_arr
        z = zarr.open(file, mode='r+')
        chunk_size = z.chunks
        z[chunk_size[0]*index:chunk_size[0]*(index+1),:] = arr
    return write_one_chunk

def zarr_file(sc, file):
    """
    Read a zarr file as an RDD of numpy arrays.
    :param sc: spark context
    :param file: file path
    :return: an RDD of numpy arrays
    """
    z = zarr.open(file, mode='r')
    ci = get_chunk_indices(z)
    chunk_indices = sc.parallelize(ci, len(ci))
    return chunk_indices.map(read_chunk(file))

def save_as_zarr_file(zarr_rdd, file):
    """
    Write an RDD of numpy arrays as a zarr file. Note that the file must already
    exist so that shape, chunk information etc is set appropriately (by the caller).
    :param zarr_rdd: n RDD of numpy arrays
    :param file: file path
    """
    zarr_rdd.foreach(write_chunk(file))

def ndarray_to_vector(arr):
    """
    Convert a numpy array to a Spark Vector.
    """
    return [Vectors.dense(row) for row in arr.tolist()]

def vectors_to_ndarray(index, vs):
    """
    Convert a partition index and list of Spark Vectors to a pair of index and numpy array.
    """
    return [(index, np.array([v.toArray() for v in vs]))]