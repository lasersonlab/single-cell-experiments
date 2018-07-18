#!/usr/bin/env bash

# Note - run commands one-by-one, not in one go (due to timing issues)

K8S_CLUSTER_NAME=ll-k8s-cluster-$USER
DASK_CLUSTER_NAME=ll-dask-cluster-$USER
ZONE=us-east1-b

gcloud container clusters create $K8S_CLUSTER_NAME \
    --num-nodes=5 \
    --machine-type=n1-standard-8 \
    --zone $ZONE \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project hca-scale

# Only needs to be run once
kubectl create clusterrolebinding cluster-admin-binding --clusterrole=cluster-admin --user=tom@lasersonlab.org

# Helm (run once per Kubernetes cluster)
kubectl --namespace kube-system create sa tiller
kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
helm init --service-account tiller

helm repo update
# replace with stable/dask when https://github.com/kubernetes/charts/tree/master/stable/dask is updated to 0.18.1
helm install --name $DASK_CLUSTER_NAME helm-charts/dask

kubectl get pods
kubectl get services

helm upgrade $DASK_CLUSTER_NAME helm-charts/dask -f scripts/dask-config.yaml

# shutdown dask
# helm delete $DASK_CLUSTER_NAME --purge

# shutdown kubernetes cluster
# gcloud container clusters delete $K8S_CLUSTER_NAME --zone $ZONE