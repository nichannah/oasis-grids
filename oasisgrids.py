#!/usr/bin/env python

from __future__ import print_function

import sys, os
import argparse
import netCDF4 as nc
import numpy as np

from esmgrids import mom_grid, nemo_grid, t42_grid, fv300_grid, oasis_grid

def check_args(args):

    err = None

    if args.model_name in ['MOM', 'NEMO']:
        if args.model_hgrid is None or \
            args.model_vgrid is None: \
            err = 'Please provide MOM or NEMO grid definition files.'

    return err

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="""
        The the model name. Supported names are:
            - MOM
            - NEMO
            - T42
            - FV300
            """)
    parser.add_argument("--grid_name", default=None, help="""
        The OASIS name for the grid being created.
        """)
    parser.add_argument("--model_hgrid", default=None, help="""
        The model horizonatal grid definition file.
        Only needed for MOM and NEMO grids""")
    parser.add_argument("--model_vgrid", default=None, help="""
        The model vertical grid definition file.
        Only needed for MOM and NEMO grids""")
    parser.add_argument("--model_mask", default=None,help="""
        The model mask file.
        Only needed for MOM and NEMO grids""")
    parser.add_argument("--grids", default="grids.nc",
                        help="The path to output OASIS grids.nc file")
    parser.add_argument("--areas", default="areas.nc",
                        help="The path to output OASIS areas.nc file")
    parser.add_argument("--masks", default="masks.nc",
                        help="The path to output OASIS masks.nc file")

    args = parser.parse_args()

    if args.grid_name is None:
        args.grid_name = args.model_name.lower()
    args.model_name = args.model_name.upper()

    err = check_args(args)
    if err is not None:
        print(err, file=sys.stderr)
        parser.print_help()
        return 1

    if args.model_name == 'MOM':
        model_grid = mom_grid.MomGrid(args.model_hgrid, args.model_vgrid,
                                      args.model_mask)
        cells = ('t', 'u')
    elif args.model_name == 'NEMO':
        model_grid = nemo_grid.NemoGrid(args.model_hgrid, args.model_vgrid,
                                        args.model_mask)
        cells = ('t', 'u', 'v')
    elif args.model_name == 'T42':
        model_grid = t42_grid.T42Grid()
        cells = ('t')
    elif args.model_name == 'FV300':
        model_grid = fv300_grid.FV300Grid()
        cells = ('t')
    else:
        assert False

    coupling_grid = oasis_grid.OasisGrid(args.grid_name, model_grid, cells)

    coupling_grid.write_grids(args.grids)
    coupling_grid.write_areas(args.areas)
    coupling_grid.write_masks(args.masks)


if __name__ == "__main__":
    sys.exit(main())
