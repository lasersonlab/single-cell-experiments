import pyspark
import sys

from anndata_spark import AnnDataRdd
from scanpy_spark import *

input = sys.argv[1]
output = sys.argv[2]

sc = pyspark.SparkContext()

adata_rdd = AnnDataRdd.from_zarr_gcs(sc, input, 'hca-scale')

recipe_zheng17(adata_rdd)

adata_rdd.write_zarr_gcs(output, (10000, adata_rdd.adata.n_vars), 'hca-scale')
