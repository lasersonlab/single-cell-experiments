import anndata as ad
import logging
import numpy_spark as np # numpy_spark includes everything in numpy, with some overrides and new functions
import numpy.testing as npt
import tempfile
import unittest

from pyspark.sql import SparkSession
from scanpy.api.pp import *
from zarr_spark import repartition_chunks

def data_file(path):
    return 'data/%s' % path


def tmp_dir():
    return tempfile.TemporaryDirectory('.zarr').name


input_file = data_file('10x-10k-subset.zarr')


class TestScanpySpark(unittest.TestCase):

    # based on https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession.builder
                .master('local[2]')
                .appName('my-local-testing-pyspark-context')
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()
        cls.sc = cls.spark.sparkContext

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.adata = ad.read_zarr(input_file) # regular anndata
        self.adata.X = self.adata.X[:] # convert to numpy array
        self.adata_rdd = ad.read_zarr(input_file) # regular anndata except for X, which we replace on the next line
        self.adata_rdd.X = np.array_rdd_zarr(self.sc, input_file + "/X")

    def get_rdd_as_array(self):
        return self.adata_rdd.X.asndarray()

    def test_log1p(self):
        log1p(self.adata_rdd)
        result = self.get_rdd_as_array()
        log1p(self.adata)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    def test_normalize_per_cell(self):
        normalize_per_cell(self.adata_rdd)
        result = self.get_rdd_as_array()
        normalize_per_cell(self.adata)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    def test_filter_cells(self):
        filter_cells(self.adata_rdd, min_genes=3)
        result = self.get_rdd_as_array()
        filter_cells(self.adata, min_genes=3)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    def test_filter_genes(self):
        filter_genes(self.adata_rdd, min_cells=2)
        result = self.get_rdd_as_array()
        filter_genes(self.adata, min_cells=2)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    # this fails when running on test data
    # def test_filter_genes_dispersion(self):
    #     filter_genes_dispersion(self.adata_rdd, flavor='cell_ranger', n_top_genes=500, log=False)
    #     result = self.get_rdd_as_array()
    #     filter_genes_dispersion(self.adata, flavor='cell_ranger', n_top_genes=500, log=False)
    #     self.assertEqual(result.shape, self.adata.shape)
    #     self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
    #     npt.assert_allclose(result, self.adata.X)

    def test_scale(self):
        scale(self.adata_rdd)
        result = self.get_rdd_as_array()
        scale(self.adata)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    # this fails when running on test data
    # def test_recipe_zheng17(self):
    #     recipe_zheng17(self.adata_rdd, n_top_genes=500)
    #     result = self.get_rdd_as_array()
    #     recipe_zheng17(self.adata, n_top_genes=500)
    #     self.assertEqual(result.shape, self.adata.shape)
    #     self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
    #     npt.assert_allclose(result, self.adata.X)

    def test_write_zarr(self):
        log1p(self.adata_rdd)
        output_file_zarr = tmp_dir()
        chunks = self.adata_rdd.X.chunks
        self.adata.write_zarr(output_file_zarr, chunks) # write metadata using regular anndata
        self.adata_rdd.X.to_zarr(output_file_zarr + "/X", chunks)
        # read back as zarr (without using RDDs) and check it is the same as self.adata.X
        adata_log1p = ad.read_zarr(output_file_zarr)
        log1p(self.adata)
        npt.assert_allclose(adata_log1p.X, self.adata.X)


if __name__ == '__main__':
    unittest.main()
