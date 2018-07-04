import pyspark

from anndata_spark import AnnDataRdd
from scanpy_spark import *

sc = pyspark.SparkContext()

adata_rdd = AnnDataRdd.from_zarr_gcs(sc, 'll-sc-data/10x/anndata_zarr/10x.zarr', 'hca-scale')

recipe_zheng17(adata_rdd)

adata_rdd.write_zarr_gcs('ll-sc-data/10x/anndata_zarr/10x-recipe.zarr', (10000, adata_rdd.adata.n_vars), 'hca-scale')
