import logging
import numpy_spark as np  # numpy_spark includes everything in numpy, with some overrides and new functions
import tempfile
import unittest
import zarr

from pyspark.sql import SparkSession


def data_file(path):
    return "data/%s" % path


def tmp_dir():
    return tempfile.TemporaryDirectory(".zarr").name


input_file = data_file("adata.csv")


class TestNumpySpark(unittest.TestCase):

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
        self.arr = np.array(
            [
                [0.0, 1.0, 0.0, 3.0, 0.0],
                [2.0, 0.0, 3.0, 4.0, 5.0],
                [4.0, 0.0, 0.0, 6.0, 7.0],
            ]
        )

        input_file_zarr = tmp_dir()
        z = zarr.open(
            input_file_zarr,
            mode="w",
            shape=self.arr.shape,
            dtype=self.arr.dtype,
            chunks=(2, 5),
        )
        z[:] = self.arr.copy()  # write as zarr, so we can read using a RDD

        self.arr_rdd = np.array_rdd_zarr(self.sc, input_file_zarr)

    def test_scalar_arithmetic(self):
        Xr = (((self.arr_rdd + 1) * 2) - 4) / 1.1
        X = (((self.arr + 1) * 2) - 4) / 1.1
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_broadcast(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Xr = self.arr_rdd + a
        X = self.arr + a
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_eq(self):
        Xr = self.arr_rdd == 0.0
        X = self.arr == 0.0
        self.assertEquals(Xr.dtype, X.dtype)
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_ne(self):
        Xr = self.arr_rdd != 0.0
        X = self.arr != 0.0
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_invert(self):
        Xr = ~(self.arr_rdd == 0.0)
        X = ~(self.arr == 0.0)
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_inplace(self):
        self.arr_rdd += 1
        self.arr += 1
        self.assertTrue(np.array_equal(self.arr_rdd.asndarray(), self.arr))

    def test_boolean_index(self):
        Xr = np.sum(self.arr_rdd, axis=1)  # sum rows
        Xr = Xr[Xr > 5]
        X = np.sum(self.arr, axis=1)  # sum rows
        X = X[X > 5]
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_subset_cols(self):
        subset = np.array([True, False, True, False, True])
        Xr = self.arr_rdd[:, subset]
        X = self.arr[:, subset]
        self.assertEquals(Xr.shape, X.shape)
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_subset_rows(self):
        subset = np.array([True, False, True])
        Xr = self.arr_rdd[subset, :]
        X = self.arr[subset, :]
        self.assertEquals(Xr.shape, X.shape)
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_log1p(self):
        log1pnps = np.log1p(self.arr_rdd).asndarray()
        log1pnp = np.log1p(self.arr)
        self.assertTrue(np.array_equal(log1pnps, log1pnp))

    def test_sum_cols(self):
        Xr = np.sum(self.arr_rdd, axis=0)
        X = np.sum(self.arr, axis=0)
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_sum_rows(self):
        Xr = np.sum(self.arr_rdd, axis=1)
        X = np.sum(self.arr, axis=1)
        self.assertTrue(np.array_equal(Xr.asndarray(), X))

    def test_mean(self):
        def mean(X):
            return X.mean(axis=0)

        meannps = mean(self.arr_rdd).asndarray()
        meannp = mean(self.arr)
        self.assertTrue(np.array_equal(meannps, meannp))

    def test_var(self):
        def var(X):
            mean = X.mean(axis=0)
            mean_sq = np.multiply(X, X).mean(axis=0)
            return mean_sq - mean ** 2

        varnps = var(self.arr_rdd).asndarray()
        varnp = var(self.arr)
        self.assertTrue(np.array_equal(varnps, varnp))


if __name__ == "__main__":
    unittest.main()
