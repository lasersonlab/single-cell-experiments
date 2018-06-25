
import h5py
import sys
import zarr
from anndata.anndata import *
import scanpy

import argparse
from os.path import splitext


parser = argparse.ArgumentParser()
parser.add_argument('input',
                    help='Path to read from')
parser.add_argument('output',
                    help='Path to write to')
parser.add_argument('--genome', '-g',
                    dest='genome',
                    required=False,
                    help="Top-level 'genome' dataset in a 10x HDF5 (.h5) file")
parser.add_argument('--chunks', '-c',
                    dest='chunks',
                    required=False,
                    help='For zarr outputs, chunk sizes for 2d arrays')


args = parser.parse_args()


input_path, input_ext = splitext(args.input)
output_path, output_ext = splitext(args.output)


if input_ext == '.h5' or input_ext == '.loom':
    if output_ext == '.zarr':
        # Convert 10x (HDF5) to Zarr
        source = h5py.File(args.input)
        zarr.tree(source)

        store = zarr.DirectoryStore(args.output)
        dest = zarr.group(store=store, overwrite=True)

        # following fails if without_attrs=False (the default), possibly related to https://github.com/h5py/h5py/issues/973
        zarr.copy_all(source, dest, log=sys.stdout, without_attrs=True)
        zarr.tree(dest)
    elif output_ext == '.h5ad':
        if not args.genome:
            raise Exception('Set --genome flag when converting from 10x HDF5 (.h5) to Anndata HDF5 (.h5ad)')
        adata = scanpy.api.read_10x_h5(args.output, genome=args.genome)
        adata.write(args.output)

elif input_ext == '.h5ad':
    adata = read_h5ad(args.input, backed=True)
    if output_ext == '.zarr':
        if not args.chunks:
            raise Exception('Provide --chunks argument for zarr output')
        [ r, c ] = [ int(d) for d in args.chunks.split(',') ]
        adata.write_zarr(args.output, chunks=(r, c))
    else:
        raise Exception('Unrecognized output extension: %s' % output_ext)
else:
    raise Exception('Unrecognized input extension: %s' % input_ext)

