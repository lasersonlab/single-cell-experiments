import math
import numpy as np
import zarr

from itertools import accumulate
from pyspark.mllib.linalg import Vectors

# Utility functions for reading and writing a Zarr array chunk; these are designed to be run as Spark tasks.
# Assume that the row lengths are small enough that the entire row fits into a Zarr chunk; in
# other words, the chunk width is the same as the row width. Also each task reads/writes a single chunk.
#
# Possible matrix operations:
# * Add or remove columns. Adjust chunk width. Easy to handle since row partitioning does not change.
# * Add or remove rows. Changes row partitioning. Simplest way to handle is to shuffle with the chunk as the key.
#   See repartition_chunks. May
#   be able to be more sophisticated with a clever Spark coalescer that can read from other partitions.
# * Matrix multiplication. Multiplying by a matrix on the right preserves partitioning, so only chunk width needs to
#   change.


def get_chunk_indices(shape, chunks):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    """
    return [
        (i, j)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
    ]


def read_zarr_chunk(arr, chunks, chunk_index):
    return arr[
        chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1),
        chunks[1] * chunk_index[1] : chunks[1] * (chunk_index[1] + 1),
    ]


def read_chunk(file):
    """
    Return a function to read a chunk by coordinates from the given file.
    """

    def read_one_chunk(chunk_index):
        """
        Read a zarr chunk specified by coordinates chunk_index=(a,b).
        """
        z = zarr.open(file, mode="r")
        return read_zarr_chunk(z, z.chunks, chunk_index)

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
        z = zarr.open(file, mode="r+")
        chunk_size = z.chunks
        z[chunk_size[0] * index : chunk_size[0] * (index + 1), :] = arr

    return write_one_chunk


def zarr_file(sc, file):
    """
    Read a zarr file as an RDD of numpy arrays.
    :param sc: spark context
    :param file: file path
    :return: an RDD of numpy arrays
    """
    z = zarr.open(file, mode="r")
    ci = get_chunk_indices(z.shape, z.chunks)
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


def repartition_chunks(sc, rows_rdd, chunks, partition_row_counts=None):
    """
    Repartition an RDD of numpy arrays with uneven row sizes so that every partition has chunks[0] rows (except for the
    last, which may have fewer).
    This function should be used before saving in Zarr format, if rows have been added or removed.
    """
    c = chunks[0]  # the chunk size for rows

    # Generate a list of offsets, so that k[i] is the number of rows before the i-th partition
    # Then turn this into a row range for each partition
    if partition_row_counts == None:

        def count_in_partition(iterator):
            return [list(iterator)[0].shape[0]]  # num rows in the matrix

        partition_row_counts = rows_rdd.mapPartitions(count_in_partition).collect()
    if all(
        [count == c for count in partition_row_counts[:-1]]
    ):  # if all except last partition have c rows...
        return (
            rows_rdd
        )  # ... then no need to shuffle, since already partitioned correctly
    k = list(accumulate([0] + partition_row_counts))
    partition_row_ranges = list(zip(k, k[1:]))
    total_rows = k[-1]
    new_num_partitions = ((total_rows - 1) // c) + 1

    def extract_partial_chunks(iterator):
        """
        For a given partition, we now know the start and end row numbers, so use that along with the new chunk size
        to break the rows into new (partial) chunks that are labelled with the new index number. Partial chunks will
        be shuffled using the new index number as key to bring together all the partial chunks for a given new index
        number.
        """
        # iterator is a single entry of ((row_start, row_end), array), where row_end is exclusive
        key, val = list(iterator)[0]
        k_i, k_i_next = key
        tuples = []
        for x in range(k_i - k_i % c, k_i_next, c):  # iterate over overlapping chunks
            start, end = max(k_i, x), min(k_i_next, x + c)
            start_offset, end_offset = start - k_i, end - k_i
            partial_chunk = val[start_offset:end_offset]
            new_index = start // c
            new_start_offset, new_end_offset = (
                start - new_index * c,
                end - new_index * c,
            )
            tuples.append(
                (new_index, ((new_start_offset, new_end_offset), partial_chunk))
            )
        return tuples

    def identity_partition_func(key):
        return key

    def combine_partial_chunks(pair):
        """
        Combine multiple non-overlapping parts of a new chunk into a single chunk.
        """
        new_index = pair[0]
        if (
            new_index == new_num_partitions - 1 and total_rows % c != 0
        ):  # last chunk has fewer than c rows
            last_chunk_rows = total_rows % c
            arr = np.zeros((last_chunk_rows, chunks[1]))
        else:
            arr = np.zeros(chunks)
        for ((new_start_offset, new_end_offset), partial_chunk) in pair[1]:
            arr[new_start_offset:new_end_offset] = partial_chunk
        return arr

    return (
        sc.parallelize(partition_row_ranges, len(partition_row_ranges))
        .zip(rows_rdd)
        .mapPartitions(extract_partial_chunks)
        .groupByKey(new_num_partitions, identity_partition_func)
        .map(combine_partial_chunks)
    )


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
