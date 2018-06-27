#!/usr/bin/env python

import h5py
import sys
import zarr
# from .anndata import anndata
from src.convert import convert
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


convert(args.input, args.output, args.genome, args.chunks)
