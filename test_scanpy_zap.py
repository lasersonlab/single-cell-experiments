import anndata as ad
import concurrent.futures
import dask.array as da
import logging
import numpy.testing as npt
import pytest
import zap.base as np  # zap includes everything in numpy, with some overrides and new functions
import zap.direct.array
import zap.executor.array
import zap.spark.array
import zarr

from pyspark.sql import SparkSession
from scanpy.api.pp import *
from scanpy.preprocessing.simple import materialize_as_ndarray


def data_file(path):
    return "data/%s" % path


input_file = data_file("10x-10k-subset.zarr")


class TestScanpy:
    @pytest.fixture(scope="module")
    def sc(self):
        # based on https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("my-local-testing-pyspark-context")
            .getOrCreate()
        )
        yield spark.sparkContext
        spark.stop()

    @pytest.fixture()
    def adata(self):
        a = ad.read_zarr(input_file)  # regular anndata
        a.X = a.X[:]  # convert to numpy array
        return a

    # "pywren" tests do not yet all pass
    @pytest.fixture(params=["direct", "executor", "spark", "dask"])
    def adata_dist(self, sc, request):
        # regular anndata except for X, which we replace on the next line
        a = ad.read_zarr(input_file)
        input_file_X = input_file + "/X"
        if request.param == "direct":
            a.X = zap.direct.array.from_zarr(input_file_X)
            yield a
        elif request.param == "executor":
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                a.X = zap.executor.array.from_zarr(executor, input_file_X)
                yield a
        elif request.param == "spark":
            a.X = zap.spark.array.from_zarr(sc, input_file_X)
            yield a
        elif request.param == "dask":
            a.X = da.from_zarr(input_file_X)
            yield a
        elif request.param == "pywren":
            import s3fs.mapping

            s3 = s3fs.S3FileSystem()
            input_file_X = s3fs.mapping.S3Map(
                "sc-tom-test-data/10x-10k-subset.zarr/X", s3=s3
            )
            executor = zap.executor.array.PywrenExecutor()
            a.X = zap.executor.array.from_zarr(executor, input_file_X)
            yield a

    def test_log1p(self, adata, adata_dist):
        log1p(adata_dist)
        result = materialize_as_ndarray(adata_dist.X)
        log1p(adata)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X)

    def test_normalize_per_cell(self, adata, adata_dist):
        normalize_per_cell(adata_dist)
        result = materialize_as_ndarray(adata_dist.X)
        normalize_per_cell(adata)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X)

    def test_filter_cells(self, adata, adata_dist):
        filter_cells(adata_dist, min_genes=3)
        result = materialize_as_ndarray(adata_dist.X)
        filter_cells(adata, min_genes=3)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X)

    def test_filter_genes(self, adata, adata_dist):
        filter_genes(adata_dist, min_cells=2)
        result = materialize_as_ndarray(adata_dist.X)
        filter_genes(adata, min_cells=2)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X)

    # # this fails when running on test data
    # def test_filter_genes_dispersion(self, adata, adata_dist):
    #     filter_genes_dispersion(adata_dist, flavor='cell_ranger', n_top_genes=500, log=False)
    #     result = materialize_as_ndarray(adata_dist.X)
    #     filter_genes_dispersion(adata, flavor='cell_ranger', n_top_genes=500, log=False)
    #     assert result.shape, adata.shape
    #     assert result.shape, (adata.n_obs, adata.n_vars)
    #     npt.assert_allclose(result, adata.X)

    def test_scale(self, adata, adata_dist):
        scale(adata_dist)
        result = materialize_as_ndarray(adata_dist.X)
        scale(adata)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X)

    # # this fails when running on test data
    # def test_recipe_zheng17(self, adata, adata_dist):
    #     recipe_zheng17(adata_dist, n_top_genes=500)
    #     result = materialize_as_ndarray(adata_dist.X)
    #     recipe_zheng17(adata, n_top_genes=500)
    #     assert result.shape, adata.shape
    #     assert result.shape, (adata.n_obs, adata.n_vars)
    #     npt.assert_allclose(result, adata.X)

    def test_write_zarr(self, adata, adata_dist):
        log1p(adata_dist)
        temp_store = zarr.TempStore()
        chunks = adata_dist.X.chunks
        print(chunks)
        # write metadata using regular anndata
        adata.write_zarr(temp_store, chunks)
        if isinstance(adata_dist.X, da.Array):
            adata_dist.X.to_zarr(temp_store.dir_path("X"))
        else:
            adata_dist.X.to_zarr(temp_store.dir_path("X"), chunks)
        # read back as zarr (without using RDDs) and check it is the same as adata.X
        adata_log1p = ad.read_zarr(temp_store)
        log1p(adata)
        npt.assert_allclose(adata_log1p.X, adata.X)
