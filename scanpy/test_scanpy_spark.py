# python3 -m venv venv3
# source venv3/bin/activate
# pip install numpy anndata pyspark

import anndata as ad
import logging
import numpy as np
import unittest

from anndata_spark import *
from pyspark.sql import SparkSession
from scanpy_spark import *

input_file = 'data/adata.csv'

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
        self.adata_rdd = AnnDataRdd.from_csv(self.sc, input_file, (2, 5))
        self.adata = ad.read_csv(input_file) # regular anndata

    def get_rdd_as_array(self):
        return np.concatenate(self.adata_rdd.rdd.collect())

    def test_log1p(self):
        log1p(self.adata_rdd)
        result = self.get_rdd_as_array()
        self.assertEquals(result.shape, (3, 5))
        log1p(self.adata)
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_filter_cells(self):
        filter_cells(self.adata_rdd, min_genes=3)
        result = self.get_rdd_as_array()
        self.assertEquals(result.shape, (2, 5))
        filter_cells(self.adata, min_genes=3)
        self.assertTrue(np.array_equal(result, self.adata.X))

    def test_filter_genes(self):
        filter_genes(self.adata_rdd, min_cells=2)
        result = self.get_rdd_as_array()
        self.assertEquals(result.shape, (3, 3))
        filter_genes(self.adata, min_cells=2)
        self.assertTrue(np.array_equal(result, self.adata.X))

if __name__ == '__main__':
    unittest.main()