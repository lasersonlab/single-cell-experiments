Zappy makes it possible to run Scanpy on Spark with relatively few changes,
however it is slower than the customized version in `scanpy_spark.py`.

```
export DATAPROC_CLUSTER_NAME=ll-cluster-$USER
export PROJECT=hca-scale
export ZONE=us-east1-d

gcloud dataproc --region us-east1 \
    clusters create $DATAPROC_CLUSTER_NAME \
    --zone $ZONE \
    --master-machine-type n1-standard-1 \
    --master-boot-disk-size 500 \
    --num-workers 20 \
    --worker-machine-type n1-standard-8 \
    --worker-boot-disk-size 500 \
    --image-version 1.2 \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project $PROJECT \
    --metadata 'CONDA_PACKAGES=numpy,PIP_PACKAGES=dask[array] gcsfs zarr git+https://github.com/lasersonlab/zappy@ncopies' \
    --initialization-actions gs://dataproc-initialization-actions/conda/bootstrap-conda.sh,gs://dataproc-initialization-actions/conda/install-conda-env.sh

gcloud dataproc jobs submit pyspark scratch/cluster-create-synthetic-data.py \
     --cluster=$DATAPROC_CLUSTER_NAME --region us-east1 --project $PROJECT
     
     
gcloud dataproc --region us-east1 clusters delete --quiet $DATAPROC_CLUSTER_NAME
```
