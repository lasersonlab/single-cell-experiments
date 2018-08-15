# pip install "dask[array]"
# pip install graphviz

# This just tests some dask operations to see if they behave as expected - it doesn't exercise any of the code in this repo.

import dask.array as da
import numpy as np
import tempfile
import unittest

def data_file(path):
    return 'data/%s' % path


def tmp_dir():
    return tempfile.TemporaryDirectory('.zarr').name


input_file = data_file('adata.csv')

class TestDask(unittest.TestCase):

    def setUp(self):
        self.arr = np.array([
            [0.0,1.0,0.0,3.0,0.0],
            [2.0,0.0,3.0,4.0,5.0],
            [4.0,0.0,0.0,6.0,7.0]
        ])
        self.arr_d = da.from_array(self.arr.copy(), chunks=(2, 5))

    def test_scalar_arithmetic(self):
        Xd = (((self.arr_d + 1) * 2) - 4) / 1.1
        X = (((self.arr + 1) * 2) - 4) / 1.1
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_broadcast(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Xd = self.arr_d + a
        X = self.arr + a
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_eq(self):
        Xd = self.arr_d == 0.0
        X = self.arr == 0.0
        self.assertEqual(Xd.dtype, X.dtype)
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_ne(self):
        Xd = self.arr_d != 0.0
        X = self.arr != 0.0
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_invert(self):
        Xd = ~(self.arr_d == 0.0)
        X = ~(self.arr == 0.0)
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_inplace(self):
        self.arr_d += 1
        self.arr += 1
        self.assertTrue(np.array_equal(self.arr_d.compute(), self.arr))

    def test_boolean_index(self):
        Xd = np.sum(self.arr_d, axis=1) # sum rows
        Xd = Xd[Xd > 5]
        X = np.sum(self.arr, axis=1) # sum rows
        X = X[X > 5]
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_subset_cols(self):
        subset = np.array([True, False, True, False, True])
        Xd = self.arr_d[:,subset]
        X = self.arr[:,subset]
        self.assertEqual(Xd.shape, X.shape)
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_subset_rows(self):
        subset = np.array([True, False, True])
        Xd = self.arr_d[subset,:]
        X = self.arr[subset,:]
        self.assertEqual(Xd.shape, X.shape)
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_log1p(self):
        log1pnps = np.log1p(self.arr_d).compute()
        log1pnp = np.log1p(self.arr)
        self.assertTrue(np.array_equal(log1pnps, log1pnp))

    def test_sum_cols(self):
        Xd = np.sum(self.arr_d, axis=0)
        X = np.sum(self.arr, axis=0)
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_sum_rows(self):
        Xd = np.sum(self.arr_d, axis=1)
        X = np.sum(self.arr, axis=1)
        self.assertTrue(np.array_equal(Xd.compute(), X))

    def test_mean(self):
        def mean(X):
            return X.mean(axis=0)
        meannps = mean(self.arr_d).compute()
        meannp = mean(self.arr)
        self.assertTrue(np.array_equal(meannps, meannp))

    def test_var(self):
        def var(X):
            mean = X.mean(axis=0)
            mean_sq = np.multiply(X, X).mean(axis=0)
            return mean_sq - mean**2
        varnps = var(self.arr_d).compute()
        varnp = var(self.arr)
        self.assertTrue(np.array_equal(varnps, varnp))

    def test_scale(self):
        def _get_mean_var(X):
            mean = X.mean(axis=0)
            mean_sq = np.multiply(X, X).mean(axis=0)
            var = (mean_sq - mean**2) * (X.shape[0]/(X.shape[0]-1))
            return mean, var
        def scale(X):
            mean, var = _get_mean_var(X)
            return (X - mean) / var
        scale(self.arr_d)
        scale(self.arr)
        # Uncomment to produce a task graph
        #scale(self.arr_d).visualize(filename='task_graph.svg')
        self.assertTrue(np.array_equal(self.arr_d.compute(), self.arr))

    def test_rechunk(self):
        arr = np.array([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0]
        ])
        arr_d = da.from_array(arr.copy(), chunks=(3, 1))
        subset = np.array([True, True, False, True, True, True, True])
        Xd = arr_d[subset,:]
        self.assertEqual(Xd.chunks, ((2, 3, 1), (1,)))
        Xd = Xd.rechunk((3, 1))
        self.assertEqual(Xd.chunks, ((3, 3), (1,)))

if __name__ == '__main__':
    unittest.main()
