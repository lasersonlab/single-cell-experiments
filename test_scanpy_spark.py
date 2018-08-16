import anndata as ad
import logging
import tempfile
import unittest

from pyspark.sql import SparkSession
from scanpy_spark import *


def data_file(path):
    return "data/%s" % path


def tmp_dir():
    return tempfile.TemporaryDirectory(".zarr").name


input_file = data_file("adata.csv")


class TestScanpySpark(unittest.TestCase):

    # based on https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (
            SparkSession.builder.master("local[2]")
            .appName("my-local-testing-pyspark-context")
            .getOrCreate()
        )

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()
        cls.sc = cls.spark.sparkContext

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.adata = ad.read_csv(input_file)  # regular anndata
        input_file_zarr = tmp_dir()
        self.adata.write_zarr(
            input_file_zarr, chunks=(2, 5)
        )  # write as zarr, so we can read using a RDD
        self.adata_rdd = AnnDataRdd.from_zarr(self.sc, input_file_zarr)

    def get_rdd_as_array(self):
        return np.concatenate(self.adata_rdd.rdd.collect())

    def test_log1p(self):
        log1p(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 5))
        log1p(self.adata)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_normalize_per_cell(self):
        normalize_per_cell(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 5))
        normalize_per_cell(self.adata)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_filter_cells(self):
        filter_cells(self.adata_rdd, min_genes=3)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (2, 5))
        filter_cells(self.adata, min_genes=3)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_filter_genes(self):
        filter_genes(self.adata_rdd, min_cells=2)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 3))
        filter_genes(self.adata, min_cells=2)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_scale(self):
        scale(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 5))
        scale(self.adata)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_write_zarr(self):
        log1p(self.adata_rdd)
        output_file_zarr = tmp_dir()
        self.adata_rdd.write_zarr(output_file_zarr, chunks=(2, 5))
        # read back as zarr (without using RDDs) and check it is the same as self.adata.X
        adata_log1p = ad.read_zarr(output_file_zarr)
        log1p(self.adata)
        self.assertTrue(np.array_equal(adata_log1p.X, self.adata.X))


if __name__ == "__main__":
    unittest.main()
