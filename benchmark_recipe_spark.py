import concurrent.futures
import csv
import os
import time

from gcloud_dataproc_util import *

project = "hca-scale"
zone = "us-east1-d"
cluster_name_fmt = "ll-cluster-tom-{}-{}"
pyspark_files = [
    "scratch/benchmark_scanpy_spark.py",
    "anndata_spark.py",
    "scanpy_spark.py",
    "zarr_spark.py",
]
bucket_name = "ll-sc-scripts"

worker_cores_variables = [2, 8, 32, 128, 512]
input_row_counts_variables = [10000, 100000, 1000000, 10000000]

input_path_fmt = "ll-sc-data-bkup/10x/anndata_zarr/10x_{}.zarr"
input_paths = [input_path_fmt.format(rows) for rows in input_row_counts_variables]

output_path_fmt = "ll-sc-data-bkup/10x/anndata_zarr_out/run={}/cores={}/input={}"
run_id = time.strftime("%Y%m%d-%H%M%S")

stop_if_duration_exceeds = 10 * 60  # 10 minutes

# upload_pyspark_files(project, bucket_name, pyspark_files)

specs = [
    (worker_cores, input_row_counts_variables, input_paths)
    for worker_cores in worker_cores_variables
]


def run_spec(spec):
    """Run the spec on a cluster."""
    worker_cores, input_row_counts, inputs = spec
    cluster_name = cluster_name_fmt.format(run_id, worker_cores)
    with dataproc_cluster(
        project, zone, cluster_name, bucket_name, worker_cores
    ) as cluster:
        results = []
        for i, input_path in enumerate(input_paths):
            input_row_count = input_row_counts[i]
            input_filename = os.path.basename(input_path)
            output_path = output_path_fmt.format(run_id, worker_cores, input_filename)

            try:
                start = time.time()
                cluster.run_pyspark_job(
                    "benchmark_scanpy_spark.py", args=[input_path, output_path]
                )
                end = time.time()
                success = True
                duration = end - start
            except Exception as e:
                print(e)
                success = False
                duration = -1

            result = {
                "worker_cores": worker_cores,
                "input_row_count": input_row_count,
                "input_path": input_path,
                "output_path": output_path,
                "success": success,
                "duration": duration,
            }
            print(result)
            results.append(result)
            if duration > stop_if_duration_exceeds:
                print("Not running more jobs on cluster {}".format(cluster_name))
                break
        return results


def save_results(results, run_id):
    with open("results/results-{}.csv".format(run_id), "w", newline="") as f:
        fieldnames = [
            "worker_cores",
            "input_row_count",
            "input_path",
            "output_path",
            "success",
            "duration",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        all_results = []
        for results in executor.map(run_spec, specs):
            print(results)
            all_results.extend(results)
        save_results(all_results, run_id)
