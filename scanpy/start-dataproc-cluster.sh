#!/usr/bin/env bash

DATAPROC_CLUSTER_NAME=tw-cluster
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

# delete cluster
# gcloud dataproc --region us-east1 clusters delete $DATAPROC_CLUSTER_NAME