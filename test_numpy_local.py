import numpy_local as np  # numpy_local includes everything in numpy, with some overrides and new functions
import tempfile
import unittest
import zarr


def data_file(path):
    return "data/%s" % path


def tmp_dir():
    return tempfile.mkdtemp(".zarr")


input_file = data_file("adata.csv")


class TestNumpyLocal(unittest.TestCase):
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

        self.arr_dist = np.ndarray_dist_local.from_zarr(input_file_zarr)

        # or test from in-memory ndarray rather than zarr
        # self.arr_dist = np.ndarray_dist_local.from_ndarray(self.arr.copy(), (2, 5))

    def test_identity(self):
        Xd = self.arr_dist
        X = self.arr
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_scalar_arithmetic(self):
        Xd = (((self.arr_dist + 1) * 2) - 4) / 1.1
        X = (((self.arr + 1) * 2) - 4) / 1.1
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_arithmetic(self):
        Xd = self.arr_dist * 2 + self.arr_dist
        X = self.arr * 2 + self.arr
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_broadcast(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Xd = self.arr_dist + a
        X = self.arr + a
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_eq(self):
        Xd = self.arr_dist == 0.0
        X = self.arr == 0.0
        self.assertEqual(Xd.dtype, X.dtype)
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_ne(self):
        Xd = self.arr_dist != 0.0
        X = self.arr != 0.0
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_invert(self):
        Xd = ~(self.arr_dist == 0.0)
        X = ~(self.arr == 0.0)
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_inplace(self):
        self.arr_dist += 1
        self.arr += 1
        self.assertTrue(np.array_equal(self.arr_dist.asndarray(), self.arr))

    def test_simple_index(self):
        Xd = self.arr_dist[0]
        X = self.arr[0]
        self.assertTrue(np.array_equal(Xd, X))

    def test_boolean_index(self):
        Xd = np.sum(self.arr_dist, axis=1)  # sum rows
        Xd = Xd[Xd > 5]
        X = np.sum(self.arr, axis=1)  # sum rows
        X = X[X > 5]
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_subset_cols(self):
        subset = np.array([True, False, True, False, True])
        Xd = self.arr_dist[:, subset]
        X = self.arr[:, subset]
        self.assertEqual(Xd.shape, X.shape)
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_subset_rows(self):
        subset = np.array([True, False, True])
        Xd = self.arr_dist[subset, :]
        X = self.arr[subset, :]
        self.assertEqual(Xd.shape, X.shape)
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_newaxis(self):
        Xd = np.sum(self.arr_dist, axis=1)[:, np.newaxis]
        X = np.sum(self.arr, axis=1)[:, np.newaxis]
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_log1p(self):
        log1pnps = np.log1p(self.arr_dist).asndarray()
        log1pnp = np.log1p(self.arr)
        self.assertTrue(np.array_equal(log1pnps, log1pnp))

    def test_sum_cols(self):
        Xd = np.sum(self.arr_dist, axis=0)
        X = np.sum(self.arr, axis=0)
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_sum_rows(self):
        Xd = np.sum(self.arr_dist, axis=1)
        X = np.sum(self.arr, axis=1)
        self.assertTrue(np.array_equal(Xd.asndarray(), X))

    def test_mean(self):
        def mean(X):
            return X.mean(axis=0)

        meannps = mean(self.arr_dist).asndarray()
        meannp = mean(self.arr)
        self.assertTrue(np.array_equal(meannps, meannp))

    def test_var(self):
        def var(X):
            mean = X.mean(axis=0)
            mean_sq = np.multiply(X, X).mean(axis=0)
            return mean_sq - mean ** 2

        varnps = var(self.arr_dist).asndarray()
        varnp = var(self.arr)
        self.assertTrue(np.array_equal(varnps, varnp))

    def test_write_zarr(self):
        output_file_zarr = tmp_dir()
        self.arr_dist.to_zarr(output_file_zarr, self.arr_dist.chunks)
        # read back as zarr (without using RDDs) and check it is the same as self.arr
        z = zarr.open(
            output_file_zarr,
            mode="r",
            shape=self.arr.shape,
            dtype=self.arr.dtype,
            chunks=(2, 5),
        )
        arr = z[:]
        self.assertTrue(np.array_equal(arr, self.arr))


if __name__ == "__main__":
    unittest.main()
