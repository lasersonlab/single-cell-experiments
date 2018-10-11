import pytest

from show_meta import show_meta


class TestShowMeta:
    @pytest.fixture()
    def h5ad_file(self):
        return "data/filtered_gene_bc_matrices.h5ad"

    def test_show_meta(self, h5ad_file):
        meta = str(show_meta(h5ad_file))
        assert (
            meta
            == """/
 ├── X
 │   ├── data (2286884,) float32
 │   ├── indices (2286884,) int32
 │   └── indptr (2701,) int32
 ├── obs (2700,) [('index', 'S16')]
 └── var (32738,) [('index', 'S19'), ('gene_ids', 'S15')]"""
        )
