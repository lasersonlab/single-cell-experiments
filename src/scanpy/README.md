# Scanpy and Spark

Experiments to make Scanpy run efficiently on Spark.

## Strategy

Scanpy use anndata for its data representation, and we'd like to explore using
zarr for the backing storage. So we need to

1. Make anndata work with zarr storage
2. Load/save anndata (backed by zarr) as Spark RDDs of numpy arrays
3. Adapt scanpy preprocessing functions to work on anndata RDDs

## Running the code

Have a look at _local.py_ for a simple demo to run locally. Requires Python 3.

## Running tests

See instructions in _test_scanpy_spark.py_.