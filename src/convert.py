import h5py
import sys
import zarr
from .anndata.anndata import read_h5ad

from scanpy.api import read_10x_h5

from os.path import splitext


def convert(input, output, genome=None, chunks=None):
    input_path, input_ext = splitext(input)
    output_path, output_ext = splitext(output)

    print('converting: %s to %s' % (input, output))

    if input_ext == '.h5' or input_ext == '.loom':
        if output_ext == '.zarr':
            # Convert 10x (HDF5) to Zarr
            source = h5py.File(input)
            zarr.tree(source)

            store = zarr.DirectoryStore(output)
            dest = zarr.group(store=store, overwrite=True)

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
            adata.write(output)

    elif input_ext == '.h5ad':
        adata = read_h5ad(input, backed='r')
        if output_ext == '.zarr':
            if not chunks:
                raise Exception('Provide --chunks argument for zarr output')
            [ r, c ] = [ int(d) for d in chunks.split(',') ]
            adata.write_zarr(output, chunks=(r, c))
        else:
            raise Exception('Unrecognized output extension: %s' % output_ext)
    else:
        raise Exception('Unrecognized input extension: %s' % input_ext)

