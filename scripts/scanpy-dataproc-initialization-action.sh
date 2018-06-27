#!/usr/bin/env bash

# This dataproc initialization action lives at gs://ll-dataproc-initialization-actions/scanpy-dataproc-initialization-action.sh
# gsutil cp scanpy-dataproc-initialization-action.sh gs://ll-dataproc-initialization-actions/scanpy-dataproc-initialization-action.sh

# Based on https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/tree/master/conda

gsutil -m cp -r gs://dataproc-initialization-actions/conda/bootstrap-conda.sh .
gsutil -m cp -r gs://dataproc-initialization-actions/conda/install-conda-env.sh .

chmod 755 ./*conda*.sh

# Install Miniconda / conda
./bootstrap-conda.sh

# Update conda root environment with specific packages in pip and conda
CONDA_PACKAGES='numpy'
PIP_PACKAGES='gcsfs scanpy zarr'

CONDA_PACKAGES=$CONDA_PACKAGES PIP_PACKAGES=$PIP_PACKAGES ./install-conda-env.sh

# Install custom anndata with zarr support
. /etc/profile.d/conda.sh
(git clone https://github.com/tomwhite/anndata; cd anndata; git checkout -b zarr origin/zarr; pip install .)