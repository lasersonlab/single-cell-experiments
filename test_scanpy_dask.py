import anndata as ad
import dask.array as da
import numpy.testing as npt
import tempfile
import unittest

from scanpy.api.pp import *

def data_file(path):
    return 'data/%s' % path


def tmp_dir():
    return tempfile.TemporaryDirectory('.zarr').name


input_file = data_file('10x-10k-subset.zarr')


class TestScanpyDask(unittest.TestCase):

    def setUp(self):
        self.adata = ad.read_zarr(input_file) # regular anndata
        self.adata.X = self.adata.X[:] # convert to numpy array
        self.adata_d = ad.read_zarr(input_file) # regular anndata except for X, which we replace on the next line
        self.adata_d.X = da.from_zarr(input_file + "/X")

    def to_ndarray(self):
        return self.adata_d.X.compute()

    def test_log1p(self):
        log1p(self.adata_d)
        result = self.to_ndarray()
        log1p(self.adata)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    def test_normalize_per_cell(self):
        normalize_per_cell(self.adata_d)
        result = self.to_ndarray()
        normalize_per_cell(self.adata)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    def test_filter_cells(self):
        filter_cells(self.adata_d, min_genes=3)
        result = self.to_ndarray()
        filter_cells(self.adata, min_genes=3)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    def test_filter_genes(self):
        filter_genes(self.adata_d, min_cells=2)
        result = self.to_ndarray()
        filter_genes(self.adata, min_cells=2)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    # this fails when running on test data
    # def test_filter_genes_dispersion(self):
    #     filter_genes_dispersion(self.adata_d, flavor='cell_ranger', n_top_genes=500, log=False)
    #     result = self.to_ndarray()
    #     filter_genes_dispersion(self.adata, flavor='cell_ranger', n_top_genes=500, log=False)
    #     self.assertEqual(result.shape, self.adata.shape)
    #     self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
    #     npt.assert_allclose(result, self.adata.X)

    def test_scale(self):
        scale(self.adata_d)
        result = self.to_ndarray()
        scale(self.adata)
        self.assertEqual(result.shape, self.adata.shape)
        self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
        npt.assert_allclose(result, self.adata.X)

    # this fails when running on test data
    # def test_recipe_zheng17(self):
    #     recipe_zheng17(self.adata_d, n_top_genes=500)
    #     result = self.to_ndarray()
    #     recipe_zheng17(self.adata, n_top_genes=500)
    #     self.assertEqual(result.shape, self.adata.shape)
    #     self.assertEqual(result.shape, (self.adata.n_obs, self.adata.n_vars))
    #     npt.assert_allclose(result, self.adata.X)

    def test_write_zarr(self):
        log1p(self.adata_d)
        output_file_zarr = tmp_dir()
        self.adata_d.X.to_zarr(output_file_zarr)
        #
        # self.adata_d.write_zarr(output_file_zarr, chunks=(2, 5))
        # # read back as zarr (without using RDDs) and check it is the same as self.adata.X
        # adata_log1p = ad.read_zarr(output_file_zarr)
        # log1p(self.adata)
        # self.assertTrue(np.array_equal(adata_log1p.X, self.adata.X))

    # def test_repartition_chunks(self):
    #     uneven_rows = [
    #         np.array([[1.], [2.], [3.], [4.]]),
    #         np.array([[5.], [6.], [7.]]),
    #         np.array([[8.], [9.], [10.], [11.]])
    #     ]
    #     uneven_rows_rdd = self.sc.parallelize(uneven_rows, len(uneven_rows))
    #     even_rows_rdd = repartition_chunks(self.sc, uneven_rows_rdd, (3, 1))
    #     even_rows = even_rows_rdd.collect()
    #     even_rows_expected = [
    #         np.array([[1.], [2.], [3.]]),
    #         np.array([[4.], [5.], [6.]]),
    #         np.array([[7.], [8.], [9.]]),
    #         np.array([[10.], [11.]])
    #     ]
    #     for i in range(len(even_rows_expected)):
    #         self.assertTrue(np.array_equal(even_rows[i], even_rows_expected[i]))
    #
    # def test_repartition_chunks_no_op(self):
    #     rows = [
    #         np.array([[1.], [2.], [3.]]),
    #         np.array([[4.], [5.], [6.]]),
    #         np.array([[7.], [8.]])
    #     ]
    #     rows_rdd = self.sc.parallelize(rows, len(rows))
    #     new_rows_rdd = repartition_chunks(self.sc, rows_rdd, (3, 1))
    #     new_rows = new_rows_rdd.collect()
    #     for i in range(len(rows)):
    #         self.assertTrue(np.array_equal(new_rows[i], rows[i]))


if __name__ == '__main__':
    unittest.main()
