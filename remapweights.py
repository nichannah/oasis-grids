#!/usr/bin/env python

import sys, os
import argparse
import netCDF4 as nc
import numpy as np

sys.path.append('./esmgrids')
from grid_factory import factory

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
    parser.add_argument("src_grid", help="File containing src grid definition")
    parser.add_argument("dest_name", help="""
        The name of the dest grid/model. Supported names are the same as
        for src_name.)
        """
    parser.add_argument("dest_grid", help="File containing dest grid def.")
    parser.add_argument("--src_mask", help="File containing src mask def.")
    parser.add_argument("--dest_mask", help="File containing dest mask def.")
    parser.add_argument("--method", default=None, help="""
        The remapping method to be used.""")

    args = parser.parse_args()

    src_grid = factory(

if __name__ == "__main__":
    sys.exit(main())
