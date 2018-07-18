#!/usr/bin/env bash

# This dataproc initialization action lives at gs://ll-dataproc-initialization-actions/scanpy-dataproc-initialization-action.sh
# gsutil cp scanpy-dataproc-initialization-action.sh gs://ll-dataproc-initialization-actions/scanpy-dataproc-initialization-action.sh

# Based on https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/tree/master/jupyter

gsutil -m cp -r gs://dataproc-initialization-actions/jupyter/jupyter.sh .

chmod 755 ./*.sh

./jupyter.sh

# Install HDF5 headers, needed for the 'tables' python package
apt-get install -y libhdf5-serial-dev

# Update conda root environment with specific packages in pip and conda
CONDA_PACKAGES='numpy'
PIP_PACKAGES='gcsfs==0.1.1 scanpy==1.2.2 zarr==2.2.0 git+https://github.com/tomwhite/anndata@zarr'
. /etc/profile.d/conda.sh
conda install $CONDA_PACKAGES
pip install $PIP_PACKAGES
