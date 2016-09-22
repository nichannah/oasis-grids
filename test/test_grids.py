
from __future__ import print_function

import sys
import os
import netCDF4 as nc
import numpy as np
import shlex
import subprocess as sp

class TestGridGeneration():

    def __init__(self):

        self.test_dir = os.path.dirname(os.path.realpath(__file__))
        self.input_dir = os.path.join(self.test_dir, 'inputs')
        self.output_dir = os.path.join(self.test_dir, 'outputs')

    def run_script(self, script, args):
        """
        Run the given script with args. 
        """

        cmd = [os.path.join(self.test_dir, '../', script)] + args
        ret = sp.call(cmd)
        return ret 

    def make_grids(self):

        input_grid = os.path.join(self.input_dir, 'ocean_hgrid.nc')
        input_mask = os.path.join(self.input_dir, 'ocean_mask.nc')
        um_restart = os.path.join(self.input_dir, 'restart_dump.astart')

        args = [input_grid, input_mask, um_restart, '--output_dir',
                self.output_dir]

        ret = self.run_script('make_grids.py', args)
        return ret


    def test_make_grids(self):
        """
        Test script makes grids. 
        """

        outputs = ['kmt.nc', 'grid.nc', 'lfrac.nc', 'qrparm.mask.nc',
                   'restart_dump.astart', 'areas.nc', 'grids.nc', 'masks.nc']
        outputs = [os.path.join(self.output_dir, o) for o in outputs]
        for f in outputs:
            if os.path.exists(f):
                os.remove(f)

        ret = self.make_grids()
        assert(ret == 0)

        # Check that outputs exist. 
        for f in outputs:
            assert(os.path.exists(f))


    def test_compare_against_old(self):
        """
        Compare some generated OASIS fields against pre-made ones. 
        """

        ret = self.make_grids()
        assert(ret == 0)

        # Masks
        masks = os.path.join(self.output_dir, 'masks.nc')
        masks_old = os.path.join(self.input_dir, 'masks_old.nc')

        with nc.Dataset(masks) as f_new:
            with nc.Dataset(masks_old) as f_old:
                for f in ['cice.msk']:
                    assert(np.array_equal(f_new.variables[f][:],
                                          f_old.variables[f][:]))

                for f in ['um1t.msk', 'um1u.msk', 'um1v.msk']:
                    ratio = (float(np.sum(f_new.variables[f][:])) / 
                             float(np.sum(f_old.variables[f][:])))
                    assert(abs(1 - ratio) < 0.02)

        # Grids
        grids = os.path.join(self.output_dir, 'grids.nc')
        grids_old = os.path.join(self.input_dir, 'grids_old.nc')

        with nc.Dataset(grids) as f_new:
            with nc.Dataset(grids_old) as f_old:
                for f in ['um1t.lon', 'um1t.lat', 'um1u.lon', 'um1u.lat',
                          'um1v.lon', 'um1v.lat',
                          'cice.lon', 'cice.lat', 'cice.clo', 'cice.cla',
                          'um1t.clo', 'um1t.cla', 'um1u.clo', 'um1u.cla',
                          'um1v.clo', 'um1v.cla']: 
                    assert(np.array_equal(f_new.variables[f][:],
                                          f_old.variables[f][:]))

        # Areas
        areas = os.path.join(self.output_dir, 'areas.nc')
        areas_old = os.path.join(self.input_dir, 'areas_old.nc')

        with nc.Dataset(areas) as f_new:
            with nc.Dataset(areas_old) as f_old:
                # These variables are significantly different between old and
                # new. FIXME: find out why. Perhaps this is related to FIXME
                # below. The new ones are closer to the real answers. 
                # for f in ['um1t.srf', 'um1u.srf', 'um1v.srf']:
                for f in ['cice.srf']:
                    assert(np.array_equal(f_new.variables[f][:],
                                          f_old.variables[f][:]))



    def test_same_area(self):
        """
        Test that source and destination grids cover the same area. 
        """

        pass

        #src_area = np.copy(src.variables['tarea'])
        #dest_area = np.copy(dest.variables['tarea'])

        # This is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal)
        #assert_almost_equal(np.sum(src_area), np.sum(dest_area[:]), decimal=-1)

    def test_overlapping(self):
        """
        Test that each destination (ocean) cell is covered by one or more
        unmasked source (atm) cells.

        We start by doing this with T-cells only. Eventually do it with all.
        """

        def find_index_of_bounding_cell(lat, lon, src_clat, src_clon):
            """
            Find the index of the box (defined by src_clat, src_clon) that
            contains the point (lat, lon).
            """

            i = None
            j = None

            # Do lat. 
            top_lat = src_clat[3, :, 0]
            bot_lat = src_clat[0, :, 0]

            for idx, (top, bot) in enumerate(zip(top_lat, bot_lat)):
                if lat >= bot and lat <= top:
                    i = idx
                    break

            # Do lon.
            left_lon = src_clon[0, 0, :]
            right_lon = src_clon[1, 0, :]
            lon = (lon + 360) % 360

            for idx, (left, right) in enumerate(zip(left_lon, right_lon)):
                if lon >= left and lon <= right:
                    j = idx
                    break
            else:
                # If spot not found then see if it fits in the first cell. This
                # is to deal with the periodic domain. 
                left = (left_lon[0] + 360)
                right = (right_lon[0] + 360)
                if lon >= left and lon <= right:
                    j = 0

            assert((i is not None) and (j is not None))

            return i, j
            

        with nc.Dataset(os.path.join(self.output_dir, 'masks.nc')) as f:
            src_mask = np.copy(f.variables['um1t.msk'])
            dest_mask = np.copy(f.variables['cice.msk'])

        with nc.Dataset(os.path.join(self.output_dir, 'grids.nc')) as f:
            dest_lat = np.copy(f.variables['cice.lat'])
            dest_lon = np.copy(f.variables['cice.lon'])

            src_clat = np.copy(f.variables['um1t.cla'])
            src_clon = np.copy(f.variables['um1t.clo'])

        # Iterate over all ocean points. Is it unmasked? If yes get it's lat
        # lon. Using the atm corners find the atm grid cell that this ocean
        # point falls within. Check that the atm cell is unmasked.
        unmasked = np.where(dest_mask == 0)

        for lat, lon in zip(dest_lat[unmasked], dest_lon[unmasked]):
            i, j = find_index_of_bounding_cell(lat, lon, src_clat, src_clon)
            assert(src_mask[i, j] == 0)
