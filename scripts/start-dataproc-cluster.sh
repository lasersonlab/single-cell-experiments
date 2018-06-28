#!/usr/bin/env bash

DATAPROC_CLUSTER_NAME=ll-cluster-$USER
gcloud dataproc --region us-east1 \
    clusters create $DATAPROC_CLUSTER_NAME \
    --master-machine-type n1-standard-1 \
    --master-boot-disk-size 500 \
    --num-workers 2 \
    --worker-machine-type n1-standard-4 \
    --worker-boot-disk-size 500 \
    --image-version 1.2 \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project hca-scale \
    --initialization-actions gs://ll-dataproc-initialization-actions/scanpy-dataproc-initialization-action.sh

# login to master
# gcloud compute ssh $DATAPROC_CLUSTER_NAME-m

# run a job
# gcloud dataproc jobs submit pyspark scratch/cluster-log1p-anndata-gcs.py \
#     --cluster=$DATAPROC_CLUSTER_NAME --region us-east1 --project hca-scale \
#     --py-files=anndata_spark.py,scanpy_spark.py


# delete cluster
# gcloud dataproc --region us-east1 clusters delete $DATAPROC_CLUSTER_NAME