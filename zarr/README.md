# Zarr and Spark

Experiments to read and write Zarr files from Spark.

## Representation

Spark's RowMatrix is used. Assume that the row lengths are small enough that the entire row fits into a Zarr chunk; in
other words, the chunk width is the same as the row width.

## Matrix operations

* Add or remove columns. Adjust chunk width. Easy to handle since row partitioning does not change.
* Add or remove rows. Changes row partitioning. Simplest way to handle is to shuffle with the chunk as the key. May
be able to be more sophisticated with a clever Spark coalescer that can read from other partitions.
* Matrix multiplication. Multiplying by a matrix on the right preserves partitioning, so only chunk width needs to
change.