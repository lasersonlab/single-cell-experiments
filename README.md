# Single Cell Experiments

Experiments to run single cell analyses efficiently at scale. Using a
combination of [Zarr], [anndata], [Scanpy], and [Apache Spark] -- and possibly
other things too.

The work in this repository is exploratory and not suitable for production.

## Overview

### [`anndata_spark.py`](anndata_spark.py)
Provides `AnnDataRdd`, an [AnnData] implementation backed by a [Spark RDD](https://spark.apache.org/docs/2.3.1/rdd-programming-guide.html#resilient-distributed-datasets-rdds)

### [`scanpy_spark.py`](scanpy_spark.py)
Some [ScanPy] functions implemented for `AnnDataRdd`s

### [`zarr_spark.py`](zarr_spark.py)
Spark convenience functions for reading and writing Zarr files as RDDs of numpy arrays.

### [`cli.py`](./cli.py)
A command-line interface for converting between various HDF5 formats (10X's, [Loom], [AnnData]) and [Zarr] equivalents:

```
# Download a 10X HDF5 file locally
wget -O files/ica_cord_blood_h5.h5 https://storage.googleapis.com/ll-sc-data/hca/immune-cell-census/ica_cord_blood_h5.h5

# Convert to .h5ad
python cli.py files/ica_cord_blood_h5.h5 files/ica_cord_blood.h5ad

# Convert to .zarr
python cli.py files/ica_cord_blood.h5ad files/ica_cord_blood.h5ad.zarr
```

`.zarr` outputs can also be written directly to `gs://` and `s3://` URLs.


## Testing

Create and activate a Python 3 virtualenv, and install the requirements:

```
python3 -m venv venv  # python 3 is required!
. venv/bin/activate
pip install -r requirements.txt
```

Install and configure Spark 2.3.1:

```
wget http://www-us.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
tar xf spark-2.3.1-bin-hadoop2.7.tgz
export SPARK_HOME=spark-2.3.1-bin-hadoop2.7
```

Run Tests:

```
pytest
```

or only run particular tests, e.g.

```
pytest -k test_log1p
```

### Troubleshooting

#### Error:

```
socket.gaierror: [Errno 8] nodename nor servname provided, or not known
```

#### Fix:
You [likely need to add a mapping for 127.0.0.1 to your `/etc/hosts`](https://stackoverflow.com/a/41231625):

```
echo '127.0.0.1    localhost' | sudo tee /etc/hosts >> /dev/null
```

#### Error:

```
  …
  File "/Users/ryan/c/hdf5-experiments/test/lib/python3.6/site-packages/pyspark/java_gateway.py", line 93, in launch_gateway
    raise Exception("Java gateway process exited before sending its port number")
Exception: Java gateway process exited before sending its port number
```

#### Fix:

```
export SPARK_LOCAL_IP=127.0.0.1
```

#### In IntelliJ:

You may need to additionally set the `PYSPARK_PYTHON` environment variable in your test configuration (to the `venv/bin/python` binary from your virtualenv above), otherwise workers will use a different/incompatible Python.

Sample configuration:

![](https://cl.ly/241w3k0d1f2d/Screen%20Shot%202018-06-27%20at%205.44.28%20PM.png)

Env vars:

![](https://cl.ly/0K0n0H132d3k/Screen%20Shot%202018-06-27%20at%205.45.12%20PM.png)

## Development

We use [Black] to enforce Python style. If you edit any Python source, you can format it with

```bash
black *.py
```

The CI will fail any patch that is not correctly formatted.

## Demos

All of the demos use Google Cloud, so you'll need an account there to run them.

### 1. Scanpy on Spark in batch mode

1. Start a five-node Dataproc cluster with the following. You'll need to change the environment
variables to ones appropriate to your account. Notice the initialization actions that install
the required Python packages on the cluster nodes.

    ```bash
    export DATAPROC_CLUSTER_NAME=ll-cluster-$USER
    export PROJECT=hca-scale
    export ZONE=us-east1-d
    
    gcloud dataproc --region us-east1 \
        clusters create $DATAPROC_CLUSTER_NAME \
        --zone $ZONE \
        --master-machine-type n1-standard-1 \
        --master-boot-disk-size 500 \
        --num-workers 5 \
        --worker-machine-type n1-standard-8 \
        --worker-boot-disk-size 500 \
        --image-version 1.2 \
        --scopes 'https://www.googleapis.com/auth/cloud-platform' \
        --project $PROJECT \
        --metadata 'CONDA_PACKAGES=numpy,PIP_PACKAGES=gcsfs scanpy zarr git+https://github.com/tomwhite/anndata@zarr' \
        --initialization-actions gs://dataproc-initialization-actions/conda/bootstrap-conda.sh,gs://dataproc-initialization-actions/conda/install-conda-env.sh
    ```

2. Run a simple Spark job. You will need to edit the output path in `scratch/cluster-log1p-anndata-gcs.py` to a
GCS bucket that you have write permissions for before running this.

    ```bash
    gcloud dataproc jobs submit pyspark scratch/cluster-log1p-anndata-gcs.py \
         --cluster=$DATAPROC_CLUSTER_NAME --region us-east1 --project $PROJECT \
         --py-files=anndata_spark.py,scanpy_spark.py,zarr_spark.py
    ```

3. Run a Scanpy recipe. Again, you'll need to change the output path.

    ```bash
    gcloud dataproc jobs submit pyspark scratch/cluster-recipe-anndata-gcs.py \
         --cluster=$DATAPROC_CLUSTER_NAME --region us-east1 --project $PROJECT \
         --py-files=anndata_spark.py,scanpy_spark.py,zarr_spark.py
    ```

4. Delete the cluster when you've finished.

    ```bash
    gcloud dataproc --region us-east1 clusters delete --quiet $DATAPROC_CLUSTER_NAME
    ```

### 2. Scanpy on Spark with Jupyter

1. Start a Dataproc cluster as before. The cluster takes longer to start since it is installing Jupyter in addition
to application packages.

    ```bash
    export DATAPROC_CLUSTER_NAME=ll-cluster-$USER
    export PROJECT=hca-scale
    export ZONE=us-east1-d
    
    gcloud dataproc --region us-east1 \
        clusters create $DATAPROC_CLUSTER_NAME \
        --zone $ZONE \
        --master-machine-type n1-standard-1 \
        --master-boot-disk-size 500 \
        --num-workers 5 \
        --worker-machine-type n1-standard-8 \
        --worker-boot-disk-size 500 \
        --image-version 1.2 \
        --scopes 'https://www.googleapis.com/auth/cloud-platform' \
        --project $PROJECT \
        --initialization-actions gs://ll-dataproc-initialization-actions/scanpy-dataproc-initialization-action.sh
    ```

2. Open a browser to the Jupyter web interface. This script opens a tunnel to the ports on the cluster.

    ```bash
    scripts/launch-jupyter-interface.sh
    ```

3. Upload and run a Jupyter notebook. From the Jupyter main page, upload `scratch/cluster-recipe-anndata-gcs.ipynb`
   and run it. You can open another browser tab to track the job progress at `http://${DATAPROC_CLUSTER_NAME}-m:8088`.

4. Delete the cluster when you've finished.

    ```bash
    gcloud dataproc --region us-east1 clusters delete --quiet $DATAPROC_CLUSTER_NAME
    ```

### 3. Scanpy on Dask with Jupyter

These instructions are based on the [Dask and Kubernetes] documentation.

1. Start a five-node Kubernetes cluster with the following. You'll need to change the environment
variables to ones appropriate to your account.

    ```bash
    export K8S_CLUSTER_NAME=ll-k8s-cluster-$USER
    export DASK_CLUSTER_NAME=ll-dask-cluster-$USER
    export PROJECT=hca-scale
    export ZONE=us-east1-b
    export EMAIL=$USER@lasersonlab.org
    
    gcloud container clusters create $K8S_CLUSTER_NAME \
        --num-nodes=5 \
        --machine-type=n1-standard-8 \
        --zone $ZONE \
        --scopes 'https://www.googleapis.com/auth/cloud-platform' \
        --project $PROJECT
    ```

2. Give your account super-user permissions.

    ```bash
    kubectl create clusterrolebinding cluster-admin-binding --clusterrole=cluster-admin --user=$EMAIL
    ```

3. Install Helm so we can use the Dask chart to install Dask.

    ```bash
    kubectl --namespace kube-system create sa tiller
    kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
    helm init --service-account tiller
    helm repo update
    ```

4. Create a vanilla Dask cluster.

    ```bash
    helm install --name $DASK_CLUSTER_NAME stable/dask
    ```

5. Upgrade the cluster to use our configuration (extra Python packages).

    ```bash
    helm upgrade $DASK_CLUSTER_NAME stable/dask -f scripts/dask-config.yaml
    ```

6. Wait until the cluster has started by checking its status with the following commands:

    ```bash
    kubectl get pods
    kubectl get services
    ```

7. When the services are running, open the Jupyter web page using the external IP reported by
   `kubectl get services` (port 80). The password is `dask`. You can open another tab to
   monitor the Dask job (look for the service whose name ends with `scheduler`). 

8. Upload and run a Jupyter notebook. From the Jupyter main page, upload `scratch/cluster-dask-recipe-anndata-gcs.ipynb`
   and run it.

9. Delete the Dask cluster when you've finished.

    ```bash
    helm delete $DASK_CLUSTER_NAME --purge
    ```

10. Delete the Kubernetes cluster.

    ```bash
    gcloud container clusters delete $K8S_CLUSTER_NAME --zone $ZONE --quiet
    ```

## People
- [Tom White](https://github.com/tomwhite/)
- [Ryan Williams](https://github.com/ryan-williams)
- [Uri Laserson](https://github.com/laserson)


[Zarr]: http://zarr.readthedocs.io/en/stable/
[anndata]: http://anndata.readthedocs.io/en/latest/
[Scanpy]: http://scanpy.readthedocs.io/en/latest/
[Apache Spark]: https://spark.apache.org/
[Loom]: http://loompy.org/
[Dask and Kubernetes]: http://dask.pydata.org/en/latest/setup/kubernetes-helm.html
[Black]: https://github.com/ambv/black