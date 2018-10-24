from contextlib import contextmanager
import os

from google.cloud import storage
import googleapiclient.discovery


@contextmanager
def dataproc_cluster(project, zone, cluster_name, bucket_name, worker_cores):
    cluster = DataprocCluster(project, zone, cluster_name, bucket_name, worker_cores)
    try:
        cluster.create()
        yield cluster
    finally:
        cluster.delete()


class DataprocCluster:
    def __init__(self, project, zone, cluster_name, bucket_name, worker_cores):
        self.project = project
        self.zone = zone
        self.region = get_region_from_zone(zone)
        self.cluster_name = cluster_name
        self.bucket_name = bucket_name
        self.worker_cores = worker_cores
        self.client = get_client()

    def create(self):
        create_cluster(
            self.client,
            self.project,
            self.zone,
            self.region,
            self.cluster_name,
            self.worker_cores,
        )
        wait_for_cluster_creation(
            self.client, self.project, self.region, self.cluster_name
        )

    def upload_pyspark_files(self, pyspark_files):
        upload_pyspark_files(self.project, self.bucket_name, pyspark_files)

    def run_pyspark_job(self, filename, args=None):
        job_id = submit_pyspark_job(
            self.client,
            self.project,
            self.region,
            self.cluster_name,
            self.bucket_name,
            filename,
            args,
        )
        return wait_for_job(
            self.client, self.project, self.region, self.cluster_name, job_id
        )

    def delete(self):
        delete_cluster(self.client, self.project, self.region, self.cluster_name)


# Based on https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/dataproc/submit_job_to_cluster.py
# and https://cloud.google.com/dataproc/docs/tutorials/python-library-example


def get_client():
    """Builds an http client authenticated with the service account
    credentials."""
    dataproc = googleapiclient.discovery.build("dataproc", "v1")
    return dataproc


def get_region_from_zone(zone):
    try:
        region_as_list = zone.split("-")[:-1]
        return "-".join(region_as_list)
    except (AttributeError, IndexError, ValueError):
        raise ValueError("Invalid zone provided, please check your input.")


def get_pyspark_file(filename):
    f = open(filename, "rb")
    return f, os.path.basename(filename)


def upload_pyspark_file(project, bucket_name, filename, file):
    """Uploads the PySpark file in this directory to the configured
    input bucket."""
    print("Uploading pyspark file to GCS")
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_file(file)


def upload_pyspark_files(project, bucket_name, pyspark_files):
    for pyspark_file in pyspark_files:
        spark_file, spark_filename = get_pyspark_file(pyspark_file)
        upload_pyspark_file(project, bucket_name, spark_filename, spark_file)


def power_of_two(n):
    return n > 0 and ((n & (n - 1)) == 0)


def get_master_machine_type(worker_cores):
    if worker_cores <= 64:
        return "n1-standard-1"
    else:
        return "n1-standard-4"


def get_worker_machine_type_and_number(worker_cores):
    assert worker_cores > 1 and power_of_two(worker_cores)
    # must have two or more worker instances
    if worker_cores == 2:
        return "n1-standard-1", 2
    elif worker_cores == 4:
        return "n1-standard-2", 2
    elif worker_cores == 8:
        return "n1-standard-4", 2
    else:
        return "n1-standard-8", worker_cores // 8


def create_cluster(dataproc, project, zone, region, cluster_name, worker_cores):
    print("Creating cluster {}...".format(cluster_name))
    zone_uri = "https://www.googleapis.com/compute/v1/projects/{}/zones/{}".format(
        project, zone
    )
    master_machine_type = get_master_machine_type(worker_cores)
    worker_machine_type, worker_num_instances = get_worker_machine_type_and_number(
        worker_cores
    )
    cluster_data = {
        "projectId": project,
        "clusterName": cluster_name,
        "config": {
            "gceClusterConfig": {
                "zoneUri": zone_uri,
                "serviceAccountScopes": [
                    "https://www.googleapis.com/auth/cloud-platform"
                ],
                "metadata": {
                    "CONDA_PACKAGES": "numpy",
                    "PIP_PACKAGES": "gcsfs scanpy zarr git+https://github.com/tomwhite/anndata@zarr",
                },
            },
            "masterConfig": {
                "numInstances": 1,
                "machineTypeUri": master_machine_type,
                "diskConfig": {"bootDiskSizeGb": 500},
            },
            "workerConfig": {
                "numInstances": worker_num_instances,
                "machineTypeUri": worker_machine_type,
                "diskConfig": {"bootDiskSizeGb": 500},
            },
            "softwareConfig": {"imageVersion": "1.2"},
            "initializationActions": [
                {
                    "executableFile": "gs://dataproc-initialization-actions/conda/bootstrap-conda.sh"
                },
                {
                    "executableFile": "gs://dataproc-initialization-actions/conda/install-conda-env.sh"
                },
            ],
        },
    }
    result = (
        dataproc.projects()
        .regions()
        .clusters()
        .create(projectId=project, region=region, body=cluster_data)
        .execute()
    )
    return result


def wait_for_cluster_creation(dataproc, project_id, region, cluster_name):
    print("Waiting for cluster creation {}...".format(cluster_name))
    while True:
        result = (
            dataproc.projects()
            .regions()
            .clusters()
            .list(projectId=project_id, region=region)
            .execute()
        )
        cluster_list = result["clusters"]
        cluster = [c for c in cluster_list if c["clusterName"] == cluster_name][0]
        if cluster["status"]["state"] == "ERROR":
            raise Exception(result["status"]["details"])
        if cluster["status"]["state"] == "RUNNING":
            print("Cluster {} created.".format(cluster_name))
            break


def list_clusters_with_details(dataproc, project, region):
    result = (
        dataproc.projects()
        .regions()
        .clusters()
        .list(projectId=project, region=region)
        .execute()
    )
    cluster_list = result["clusters"]
    for cluster in cluster_list:
        print("{} - {}".format(cluster["clusterName"], cluster["status"]["state"]))
    return result


def submit_pyspark_job(
    dataproc, project, region, cluster_name, bucket_name, filename, args=None
):
    """Submits the Pyspark job to the cluster, assuming `filename` has
    already been uploaded to `bucket_name`"""
    job_details = {
        "projectId": project,
        "job": {
            "placement": {"clusterName": cluster_name},
            "pysparkJob": {
                "mainPythonFileUri": "gs://{}/{}".format(bucket_name, filename),
                "pythonFileUris": [
                    "gs://ll-sc-scripts/anndata_spark.py",
                    "gs://ll-sc-scripts/scanpy_spark.py",
                    "gs://ll-sc-scripts/zarr_spark.py",
                ],
            },
        },
    }
    if args is not None:
        job_details["job"]["pysparkJob"]["args"] = args
    result = (
        dataproc.projects()
        .regions()
        .jobs()
        .submit(projectId=project, region=region, body=job_details)
        .execute()
    )
    job_id = result["reference"]["jobId"]
    print("Submitted job ID {} to cluster {}".format(job_id, cluster_name))
    return job_id


def wait_for_job(dataproc, project, region, cluster_name, job_id):
    print("Waiting for job {} on cluster {} to finish...".format(job_id, cluster_name))
    while True:
        result = (
            dataproc.projects()
            .regions()
            .jobs()
            .get(projectId=project, region=region, jobId=job_id)
            .execute()
        )
        # Handle exceptions
        if result["status"]["state"] == "ERROR":
            print("Job {} on cluster {} failed.".format(job_id, cluster_name))
            raise Exception(result["status"]["details"])
        elif result["status"]["state"] == "DONE":
            print("Job {} on cluster {} finished.".format(job_id, cluster_name))
            return result


def delete_cluster(dataproc, project, region, cluster_name):
    print("Deleting cluster {}".format(cluster_name))
    result = (
        dataproc.projects()
        .regions()
        .clusters()
        .delete(projectId=project, region=region, clusterName=cluster_name)
        .execute()
    )
    return result
