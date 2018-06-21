
import h5py
import sys
import zarr

import argparse
from os.path import splitext

parser = argparse.ArgumentParser()
parser.add_argument('input',
                    help='Path to read from')
parser.add_argument('output',
                    help='Path to write to')

args = parser.parse_args()

input_path, input_ext = splitext(args.input)
output_path, output_ext = splitext(args.output)

if input_ext == '.h5' or input_ext == '.loom':
    # Convert 10x (HDF5) to Zarr
    source = h5py.File(args.input)
    zarr.tree(source)

    store = zarr.DirectoryStore(args.output)
    dest = zarr.group(store=store, overwrite=True)
    # following fails if without_attrs=False (the default), possibly related to https://github.com/h5py/h5py/issues/973
    zarr.copy_all(source, dest, log=sys.stdout, without_attrs=True)
    zarr.tree(dest)

# elif input_ext == '.h5ad':
#     pass
else:
    raise Exception('Unrecognized input extension: %s' % input_ext)

