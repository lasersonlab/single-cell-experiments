# # On a GCP ubuntu node n1-standard-16 (16 vCPUs, 60 GB memory),
# # or a n1-ultramem-40 - 961GB mem, 40GB disk to generate the 10e6 subset
# sudo apt-get install -y python3-pip python3-tk git
# pip3 install gcsfs numpy scanpy zarr

# Copy 10x data locally
# gsutil -u hca-scale cp gs://ll-sc-data/10x/1M_neurons_filtered_gene_bc_matrices_h5.h5 1M_neurons_filtered_gene_bc_matrices_h5.h5

# python3

import gcsfs.mapping
import scanpy.api

gcs = gcsfs.GCSFileSystem('hca-scale', token='cloud')
store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr/10x.zarr', gcs=gcs)

adata = scanpy.api.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')
chunks=(10000, adata.n_vars)

adata.write_zarr(store, chunks)

# write subsets
for n in range(6, 7):
    N = pow(10, n)
    print("Writing subset of size {}".format(N))
    store = gcsfs.mapping.GCSMap('ll-sc-data-bkup/10x/anndata_zarr/10x_{}.zarr'.format(N), gcs=gcs)
    a = adata[:N, ]
    a.write_zarr(store, chunks)
