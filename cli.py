#!/usr/bin/env python

from src.convert import convert

import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('input',
                    help='Path to read from')
parser.add_argument('output',
                    default=None,
                    nargs='?',
                    help='Path to write to; defaults to "%s.zarr" % <input>')
parser.add_argument('--genome', '-g',
                    dest='genome',
                    required=False,
                    help="Top-level 'genome' dataset in a 10x HDF5 (.h5) file")
parser.add_argument('--force', '-f',
                    dest='overwrite',
                    type=bool,
                    default=False,
                    required=False,
                    help="If true, overwrite any existing output file that exists (if False, raise an Exception in this case)")
parser.add_argument('--chunks', '-c',
                    dest='chunks',
                    required=False,
                    default='16m',
                    help='For zarr outputs, chunk sizes for 2d arrays')


args = parser.parse_args()


def bytes_value(s):
    scales = {
        '' : 1,
        'k': 1024,
        'm': 1024 * 1024,
        'g': 1024 * 1024 * 1024,
    }

    m = re.match('(\d+)(?:([kmg]?)b?)', s.lower())

    if not m:
        raise Exception('Invalid bytes string: %s' % s)

    return int(m.group(1)) * scales[m.group(2)]


convert(
    args.input,
    args.output if args.output else (args.input + '.zarr'),
    genome=args.genome,
    chunk_size=bytes_value(args.chunks),
    overwrite=args.overwrite
)
