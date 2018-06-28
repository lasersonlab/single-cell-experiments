# python3 -m venv venv3
# source venv3/bin/activate
# pip install numpy anndata
# export PYSPARK_PYTHON=$(pwd)/venv3/bin/python
# ~/sw/spark-2.2.1-bin-hadoop2.7/bin/pyspark

# Use Spark to emulate a very basic Scanpy function.

import shutil

from scanpy_spark import *

input_file = '/tmp/adata.csv'
shutil.copy('data/adata.csv', input_file)

spark = SparkSession \
    .builder \
    .appName("scanpy-demo") \
    .getOrCreate()
sc = spark.sparkContext

adata_rdd = AnnDataRdd.from_csv(sc, input_file, (2, 5))

def pr(x):
    print(x)

adata_rdd.rdd.foreach(pr)

# Call log1p function which emulates scanpy
log1p(adata_rdd) # updates in place
adata_rdd.rdd.foreach(pr)

# filter cells
adata_rdd = AnnDataRdd.from_csv(sc, input_file, (2, 5))
adata_rdd.rdd.foreach(pr)
filter_cells(adata_rdd, min_genes=3)
adata_rdd.rdd.foreach(pr)

# filter genes
adata_rdd = AnnDataRdd.from_csv(sc, input_file, (2, 5))
adata_rdd.rdd.foreach(pr)
filter_genes(adata_rdd, min_cells=2)
adata_rdd.rdd.foreach(pr)

