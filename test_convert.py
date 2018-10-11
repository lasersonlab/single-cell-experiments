import pytest

from anndata import read_zarr

from convert import convert


class TestConvert:
    @pytest.fixture()
    def h5_file(self):
        return "data/filtered_gene_bc_matrices_h5.h5"

    @pytest.fixture()
    def h5ad_file(self):
        return "data/filtered_gene_bc_matrices.h5ad"

    def test_10x_h5_to_zarr(self, h5_file, tmpdir):
        p = tmpdir.join("filtered_gene_bc_matrices.zarr")
        input = h5_file
        output = str(p)
        convert(input, output)

        # read back and check a few things
        adata = read_zarr(output)
        assert adata.X.shape == (34, 343)
        assert adata.obs.shape == (34, 0)
        assert adata.var.shape == (343, 1)

    def test_h5ad_to_zarr(self, h5ad_file, tmpdir):
        p = tmpdir.join("filtered_gene_bc_matrices.zarr")
        input = h5ad_file
        output = str(p)
        convert(input, output)

        # read back and check a few things
        adata = read_zarr(output)
        assert adata.X.shape == (2700, 32738)
        assert adata.obs.shape == (2700, 0)
        assert adata.var.shape == (32738, 1)
