#!/usr/bin/env python

from __future__ import print_function

import sys, os
import argparse
import netCDF4 as nc
import numpy as np

from esmgrids.mom_grid import MomGrid
from esmgrids.nemo_grid import NemoGrid
from esmgrids.t42_grid import T42Grid
from esmgrids.fv300_grid import FV300Grid
from esmgrids.core2_grid import Core2Grid
from esmgrids.jra55_grid import Jra55Grid
from esmgrids.oasis_grid import OasisGrid

def check_args(args):

    err = None

    if args.model_name in ['MOM', 'NEMO']:
        if args.model_hgrid is None or args.model_mask is None:
            err = 'Please provide MOM or NEMO grid definition and mask files.'

    return err

def check_file_exist(files):

    err = None

    for f in files:
        if f is not None and not os.path.exists(f):
            err = "Can't find input file {}.".format(f)

    return err

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="""
        The the model name. Supported names are:
            - MOM   # 1, 0.25 and 0.1 degree MOM ocean
            - NEMO  # ocean
            - SPE   # T42 spectral atmos
            - FVO   # 2 degree atmos
            - CORE2 # CORE2 atmosphere
            - JRA55 # JRA55 atmosphere
            """)
    parser.add_argument("--grid_name", default=None, help="""
        The OASIS name for the grid being created.
        """)
    parser.add_argument("--model_hgrid", default=None, help="""
        The model horizonatal grid definition file.
        Only needed for MOM and NEMO grids""")
    parser.add_argument("--model_mask", default=None, help="""
        The model mask file.
        Only needed for MOM and NEMO grids""")
    parser.add_argument("--model_cols", type=int, default=128, help="""
        Number of model columns
        Only needed for atmospheric grids""")
    parser.add_argument("--model_rows", type=int, default=64, help="""
        Number of model rows
        Only needed for atmospheric grids""")
    parser.add_argument("--grids", default="grids.nc",
                        help="The path to output OASIS grids.nc file")
    parser.add_argument("--areas", default="areas.nc",
                        help="The path to output OASIS areas.nc file")
    parser.add_argument("--masks", default="masks.nc",
                        help="The path to output OASIS masks.nc file")

    args = parser.parse_args()

    # The model name needs to be a certain length because OASIS requires that
    # grid names are exactly 4 characters long plus postfix.
    assert len(args.model_name) >= 3

    if args.grid_name is None:
        args.grid_name = args.model_name.lower()
    args.model_name = args.model_name.upper()

    err = check_args(args)
    if err is not None:
        print(err, file=sys.stderr)
        parser.print_help()
        return 1

    err = check_file_exist([args.model_hgrid, args.model_mask])
    if err is not None:
        print(err, file=sys.stderr)
        parser.print_help()
        return 1

    if args.model_name == 'MOM':
        model_grid = MomGrid.fromfile(args.model_hgrid,
                                      mask_file=args.model_mask)
        cells = ('t', 'u')
    elif args.model_name == 'CICE':
        model_grid = CiceGrid.fromfile(args.model_hgrid,
                                       mask_file=args.model_mask)
        cells = ('t', 'u')
    elif args.model_name == 'NEMO':
        model_grid = NemoGrid(args.model_hgrid, mask_file=args.model_mask)
        cells = ('t', 'u', 'v')
    elif args.model_name == 'SPE':
        num_lons = args.model_cols
        num_lats = args.model_rows
        model_grid = T42Grid(num_lons, num_lats, 1, args.model_mask,
                                      description='Spectral')
        cells = ('t')
    elif args.model_name == 'FVO':
        num_lons = args.model_cols
        num_lats = args.model_rows
        model_grid = FV300Grid(num_lons, num_lats, 1, args.model_mask,
                                          description='FV')
        cells = ('t')
    elif args.model_name == 'CORE2':
        model_grid = Core2Grid(192, 94, 1, args.model_mask,
                               description=args.model_name)
        cells = ('t')
    elif args.model_name == 'JRA55':
        model_grid = Jra55Grid(360, 180, 1, args.model_mask,
                               description=args.model_name)
        cells = ('t')
    else:
        assert False

    coupling_grid = OasisGrid(args.grid_name, model_grid, cells)

    coupling_grid.write_grids(args.grids)
    coupling_grid.write_areas(args.areas)
    coupling_grid.write_masks(args.masks)

if __name__ == "__main__":
    sys.exit(main())
