from gcsfs import GCSFileSystem
from gcsfs.mapping import GCSMap
from h5py import Dataset, File, Group
from s3fs import S3FileSystem
from s3fs.mapping import S3Map
import json
import os
import sys
import zarr

from scanpy.api import read_10x_h5, read_h5ad, read_loom

from os import path
from os.path import splitext

import re
from urllib.parse import urlparse


def gcpProject():
    file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not file:
        raise Exception("Set $GOOGLE_APPLICATION_CREDENTIALS env var")
    if path.exists(file):
        with open(file) as fd:
            return json.load(fd)["project_id"]
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS file doesn't exist: %s" % file)


_gcsFileSystem = None


def gcsFileSystem():
    global _gcsFileSystem
    if not _gcsFileSystem:
        project = gcpProject()
        _gcsFileSystem = GCSFileSystem(project=project)
    return _gcsFileSystem


def exists(uri):
    p = urlparse(uri)
    if p.scheme == "gs":
        gcsFileSystem().exists(uri)
    elif p.scheme == "s3":
        fs = S3FileSystem()
        fs.exists(uri)
    elif not p.scheme:
        os.path.exists(uri)
    else:
        raise Exception("Unrecognized scheme %s in URL %s" % (p.scheme, uri))


def make_store(path):
    m = re.match("^gc?s://", path)
    if m:
        return GCSMap(path[len(m.group(0)) :], gcs=gcsFileSystem())

    if path.startswith("s3://"):
        s3 = S3FileSystem()
        return S3Map(path[len("s3://") :], s3=s3)

    return zarr.DirectoryStore(path)


def suggested_chunks(chunk_size, arr, axis=0):
    elems_per_main_axis_entry = arr.size // arr.shape[axis]
    size_per_main_axis_entry = elems_per_main_axis_entry * arr.dtype.itemsize
    main_axis_entries_per_chunk = chunk_size // size_per_main_axis_entry
    chunks = list(arr.shape)
    chunks[axis] = min(chunks[axis], main_axis_entries_per_chunk)
    return tuple(chunks)


def read_anndata(input, genome=None):
    _, input_ext = splitext(input)
    if input_ext == ".h5":
        if not genome:
            keys = list(File(input, "r").keys())
            if len(keys) == 1:
                genome = keys[0]
            else:
                raise Exception(
                    "Set --genome flag when converting from 10x HDF5 (.h5) to Anndata HDF5 (.h5ad); top-level groups in file %s: %s"
                    % (input, ",".join(keys))
                )
        return read_10x_h5(input, genome=genome)
    elif input_ext == ".h5ad":
        return read_h5ad(input)
    elif input_ext == ".loom":
        # reads the whole dataset in memory!
        return read_loom(input)
    else:
        raise Exception("Unrecognized input extension: %s" % input_ext)


def convert(input, output, chunk_size=16 * 1024 * 1024, genome=None, overwrite=False):
    if exists(output) and not overwrite:
        raise Exception(
            "Output path already exists: %s; use --overwrite/-f to overwrite" % output
        )

    print("converting: %s to %s" % (input, output))

    adata = read_anndata(input, genome=genome)
    _, output_ext = splitext(output)
    if output_ext == ".zarr":
        chunks = suggested_chunks(chunk_size, adata.X)
        store = make_store(output)
        adata.write_zarr(store, chunks)
    elif output_ext == ".h5ad":
        adata.write(output)
    else:
        raise Exception("Unrecognized output extension: %s" % output_ext)
