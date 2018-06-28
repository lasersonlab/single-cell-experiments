import gcsfs
import h5py
from os.path import getsize
import s3fs
import sys
import zarr
from anndata import read_h5ad

from scanpy.api import read_10x_h5

from os.path import splitext

import re

def make_store(path):
    m = re.match('^gc?s://', path)
    if m:
        gcs = gcsfs.GCSFileSystem()
        return gcsfs.mapping.GCSMap(path[len(m.group(0)):], gcs=gcs)

    if path.startswith('s3://'):
        s3 = s3fs.S3FileSystem()
        return s3fs.mapping.S3Map(path[len('s3://')], s3=s3)

    return zarr.DirectoryStore(path)


def convert(
        input,
        output,
        chunk_size=16 * 1024 * 1024,
        genome=None,
        overwrite=False
):
    input_path, input_ext = splitext(input)
    output_path, output_ext = splitext(output)

    print('converting: %s to %s' % (input, output))

    if input_ext == '.h5' or input_ext == '.loom':
        if output_ext == '.zarr':
            # Convert 10x (HDF5) to Zarr
            source = h5py.File(input)
            zarr.tree(source)

            store = zarr.DirectoryStore(output)
            dest = zarr.group(store=store, overwrite=overwrite)

            # following fails if without_attrs=False (the default), possibly related to https://github.com/h5py/h5py/issues/973
            zarr.copy_all(source, dest, log=sys.stdout, without_attrs=True)
            zarr.tree(dest)
        elif output_ext == '.h5ad':
            if not genome:
                keys = list(h5py.File(input).keys())
                if len(keys) == 1:
                    genome = keys[0]
                else:
                    raise Exception(
                        'Set --genome flag when converting from 10x HDF5 (.h5) to Anndata HDF5 (.h5ad); top-level groups in file %s: %s'
                        % (input, ','.join(keys))
                    )
            adata = read_10x_h5(input, genome=genome)

            # TODO: respect overwrite flag
            adata.write(output)

    elif input_ext == '.h5ad':
        adata = read_h5ad(input, backed='r')
        (r, c) = adata.shape
        chunks = (getsize(input) - 1) / chunk_size + 1
        chunk_size = (r - 1) / chunks + 1
        if output_ext == '.zarr':
            print('converting %s (%dx%d) to %s in %d chunks (%d rows each)' % (input, r, c, output, chunks, chunk_size))

            # TODO: respect overwrite flag

            adata.write_zarr(
                make_store(output),
                chunks=(chunk_size, c)
            )
        else:
            raise Exception('Unrecognized output extension: %s' % output_ext)
    else:
        raise Exception('Unrecognized input extension: %s' % input_ext)

