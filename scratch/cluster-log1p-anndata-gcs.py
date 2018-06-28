#
import pyspark

from anndata_spark import AnnDataRdd
from scanpy_spark import *

sc = pyspark.SparkContext()

adata_rdd = AnnDataRdd.from_zarr_gcs(sc, 'll-sc-data/10x/anndata_zarr/10x.zarr', 'hca-scale')

log1p(adata_rdd) # updates in place

adata_rdd.write_zarr_gcs('ll-sc-data/10x/anndata_zarr/10x-log1p.zarr', (10000, 27998), 'hca-scale')