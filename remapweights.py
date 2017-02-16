#!/usr/bin/env python

import sys, os
import sh
import shutil
import argparse
import netCDF4 as nc
import numpy as np
import numba
import tempfile
import subprocess as sp

sys.path.append('./esmgrids')
from grid_factory import factory

def create_regrid_weights(src_grid, dest_grid, method='conserve',
                          unmasked_src=True, unmasked_dest=False):

    _, src_grid_scrip = tempfile.mkstemp(suffix='.nc')
    _, dest_grid_scrip = tempfile.mkstemp(suffix='.nc')
    _, regrid_weights = tempfile.mkstemp(suffix='.nc')

    if unmasked_src:
        src_grid.write_scrip(src_grid_scrip,
                            mask=np.zeros_like(src_grid.mask_t, dtype=int))
    else:
        src_grid.write_scrip(src_grid_scrip)

    if unmasked_dest:
        dest_grid.write_scrip(dest_grid_scrip,
                              mask=np.zeros_like(dest_grid.mask_t, dtype=int))
    else:
        dest_grid.write_scrip(dest_grid_scrip)

    mpirun = []
    if sh.which('mpirun') is not None:
        import multiprocessing as mp
        mpirun = ['mpirun', '-np', str(mp.cpu_count() // 2)]
    try:
        cmd = mpirun + ['ESMF_RegridWeightGen',
                        '-s', src_grid_scrip,
                        '-d', dest_grid_scrip, '-m', method,
                        '-w', regrid_weights]
        sp.check_output(cmd)
    except sp.CalledProcessError as e:
        print("Error: ESMF_RegridWeightGen failed ret {}".format(e.returncode),
              file=sys.stderr)
        print(e.output, file=sys.stderr)
        log = 'PET0.RegridWeightGen.Log'
        if os.path.exists(log):
            print('Contents of {}:'.format(log), file=sys.stderr)
            with open(log) as f:
                print(f.read(), file=sys.stderr)
        return None

    os.remove(src_grid_scrip)
    os.remove(dest_grid_scrip)

    return regrid_weights


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("src_name", help="""
        The name of the src grid/model. Supported names are:
            MOM   (1, 0.25 and 0.1 degree MOM ocean),
            NEMO  (ocean),
            SPE   (T42 spectral atmos),
            FVO   (2 degree atmos),
            CORE2 (CORE2 atmosphere),
            JRA55 (JRA55 atmosphere),
            """)
    parser.add_argument("dest_name", help="""
        The name of the dest grid/model. Supported names are the same as
        for src_name.
        """)
    parser.add_argument("--src_grid", default=None,
                        help="File containing src grid definition")
    parser.add_argument("--dest_grid", default=None,
                        help="File containing dest grid def.")
    parser.add_argument("--src_mask", default=None,
                        help="File containing src mask def.")
    parser.add_argument("--dest_mask", default=None,
                        help="File containing dest mask def.")
    parser.add_argument("--method", default='bilinear', help="""
        The remapping method to be used.""")
    parser.add_argument("--output", default=None, help="""
        Name of the output file.""")

    args = parser.parse_args()

    if args.output is None:
        args.output = '{}_{}_{}_rmp.nc'.format(args.src_name,
                                               args.dest_name, args.method)

    src_grid = factory(args.src_name, args.src_grid, args.src_mask)
    dest_grid = factory(args.dest_name, args.dest_grid, args.dest_mask)

    regrid_weights = create_regrid_weights(src_grid, dest_grid,
                                           method=args.method,
                                           unmasked_src=True,
                                           unmasked_dest=True)

    if regrid_weights is None:
        return 1
    else:
        shutil.move(regrid_weights, args.output)

    return 0

if __name__ == "__main__":
    sys.exit(main())
