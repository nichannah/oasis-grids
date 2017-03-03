#!/usr/bin/env python

import sys, os
import sh
import shutil
import shlex
import argparse
import netCDF4 as nc
import numpy as np
import numba
import tempfile
import subprocess as sp

from grid_factory import factory

def convert_to_scrip_output(weights):

    _, new_weights = tempfile.mkstemp(suffix='.nc')
    # FIXME: So that ncrename doesn't prompt for overwrite.
    os.remove(new_weights)

    cmd = 'ncrename -d n_a,src_grid_size -d n_b,dst_grid_size -d n_s,num_links -d nv_a,src_grid_corners -d nv_b,dst_grid_corners -v yc_a,src_grid_center_lat -v yc_b,dst_grid_center_lat -v xc_a,src_grid_center_lon -v xc_b,dst_grid_center_lon -v yv_a,src_grid_corner_lat -v xv_a,src_grid_corner_lon -v yv_b,dst_grid_corner_lat -v xv_b,dst_grid_corner_lon -v mask_a,src_grid_imask -v mask_b,dst_grid_imask -v area_a,src_grid_area -v area_b,dst_grid_area -v frac_a,src_grid_frac -v frac_b,dst_grid_frac -v col,src_address -v row,dst_address {} {}'.format(weights, new_weights)

    try:
        sp.check_output(shlex.split(cmd))
    except sp.CalledProcessError as e:
        print(e.output, file=sys.stderr)

    # Fix the dimension of the remap_matrix.
    with nc.Dataset(weights) as f_old, nc.Dataset(new_weights, 'r+') as f_new:
        remap_matrix = f_new.createVariable('remap_matrix', 'f8', ('num_links', 'num_wgts'))
        remap_matrix[:, 0] = f_old.variables['S'][:]

    os.remove(weights)

    return new_weights

def create_weights(src_grid, dest_grid, method='conserve',
                   ignore_unmapped=False,
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

    if ignore_unmapped:
        ignore_unmapped = ['--ignore_unmapped']
    else:
        ignore_unmapped = []

    mpirun = []
    if sh.which('mpirun') is not None:
        import multiprocessing as mp
        mpirun = ['mpirun', '-np', str(mp.cpu_count() // 2)]

    my_dir = os.path.dirname(os.path.realpath(__file__))
    esmf = os.path.join(my_dir, 'contrib', 'bin', 'ESMF_RegridWeightGen')
    if not os.path.exists(esmf):
        esmf = 'ESMF_RegridWeightGen'

    try:
        cmd = mpirun + [esmf] + [
                        '-s', src_grid_scrip,
                        '-d', dest_grid_scrip, '-m', method,
                        '-w', regrid_weights] + ignore_unmapped
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
    parser.add_argument("--ignore_unmapped", action='store_true',
                        default=False,
                        help="Ignore unmapped destination points. Don't use.")
    parser.add_argument("--output", default=None,
                        help="Name of the output file.")
    parser.add_argument("--output_convention", default='NCAR-CSM', help="""
                         The variable name output convention to use.
                         Can be NCAR-CSM or SCRIP.""")

    args = parser.parse_args()

    assert args.output_convention == 'NCAR-CSM' or \
           args.output_convention == 'SCRIP'

    if args.output is None:
        args.output = '{}_{}_{}_rmp.nc'.format(args.src_name,
                                               args.dest_name, args.method)

    src_grid = factory(args.src_name, args.src_grid, args.src_mask)
    dest_grid = factory(args.dest_name, args.dest_grid, args.dest_mask)

    weights = create_weights(src_grid, dest_grid, method=args.method,
                             ignore_unmapped=args.ignore_unmapped,
                              unmasked_src=True, unmasked_dest=True)

    if weights is None:
        return 1

    if args.output_convention == 'SCRIP':
        weights = convert_to_scrip_output(weights)

    shutil.move(weights, args.output)

    return 0

if __name__ == "__main__":
    sys.exit(main())
