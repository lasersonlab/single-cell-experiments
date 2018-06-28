# python3 -m venv venv3
# source venv3/bin/activate
# pip install numpy pyspark zarr src/anndata
# python test_scanpy_spark.py


import src.anndata.anndata as ad
import logging
import numpy as np
import unittest

from src.scanpy.anndata_spark import AnnDataRdd
from pyspark.sql import SparkSession
from src.scanpy.scanpy_spark import *

def data_file(path):
    return 'src/scanpy/data/%s' % path

input_file = data_file('adata.csv')

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
        self.adata = ad.read_csv(input_file) # regular anndata
        input_file_zarr = data_file('anndata.zarr')
        self.adata.write_zarr(input_file_zarr, chunks=(2, 5)) # write as zarr, so we can read using a RDD
        self.adata_rdd = AnnDataRdd.from_zarr(self.sc, input_file_zarr)


    def get_rdd_as_array(self):
        return np.concatenate(self.adata_rdd.rdd.collect())

    def test_log1p(self):
        log1p(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 5))
        log1p(self.adata)
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_normalize_per_cell(self):
        normalize_per_cell(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 5))
        normalize_per_cell(self.adata)
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_filter_cells(self):
        filter_cells(self.adata_rdd, min_genes=3)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (2, 5))
        filter_cells(self.adata, min_genes=3)
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_filter_genes(self):
        filter_genes(self.adata_rdd, min_cells=2)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 3))
        filter_genes(self.adata, min_cells=2)
        self.assertTrue(np.array_equal(result, self.adata.X))
    def test_scale(self):
        scale(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEqual(result.shape, (3, 5))
        scale(self.adata)
        self.assertTrue(np.array_equal(result, self.adata.X))

if __name__ == '__main__':
    unittest.main()
