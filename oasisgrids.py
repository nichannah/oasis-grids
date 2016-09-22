#!/usr/bin/env python

from __future__ import print_function

import sys, os
import argparse
import netCDF4 as nc
import numpy as np
import shlex
import shutil
import subprocess as sp
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from regridder import Regridder
from mkancil import Mkancil
from remask_um_restart import remask

EARTH_AREA = 510072000e6

"""
What this program does: 

Given ocean grid, mask, and the resolution of the atmosphere grid this program
will generate:

1) The CICE grid and land mask. 
2) The OASIS config files: masks.nc, areas.nc, grids.nc
3) The atmosphere land sea fraction. 

The atmosphere grid is assumed to be regular. 
"""

def normalise_lons(lons):
    """
    Make a copy of lons that is between -180 and 180.
    """

    lons_n = np.copy(lons)

    lons_n[lons_n > 180] = lons_n[lons_n > 180] - 360
    lons_n[lons_n < -180] = lons_n[lons_n < -180] + 360
    return lons_n


def oasis_to_2d_corners(input_clons, input_clats):
    """
    Change from an oasis corners convention (3D) to 2D convention. Also
    changes longitudes to be from -180 to 180.

    The input_array has shape (4, rows, columns) where the first dimension is
    the corner number with 0 being lower left and increasing in
    counter-clockwise dimension:

    3---2
    |   |
    0---1

    From this see that the top corners of one grid cell (3, 2) will be the
    bottom corners (0, 1) of the grid cell above. The conversion will not
    work if this assumption does not hold.

    3---2 3---2
    |   | |   |
    0---1 0---1
    3---2 3---2
    |   | |   |
    0---1 0---1

    The output array has shape (rows + 1, columns). It contains all the 0
    corners as well as the top row 3 corners. An extra column is not needed
    because it is periodic, so the the right hand corners and the right-most
    cell are in fact the left-most corners. 
    """

    def check_array(array):
        """
        Ensure that input array supports assumption that there are no 'gaps'
        between cells. These are the assumptions explained above. 
        """

        # 0 and 3 corners have the same longitude along rows.
        assert(np.sum(array[0, 1:, :] == array[3, 0:-1, :]) ==
               array[0, 1:, :].size)
        # 1 and 2 corners have the same longitude along rows.
        assert(np.sum(array[1, 1:, :] == array[2, 0:-1, :]) ==
               array[1, 1:, :].size)
        # 0 and 1 corners have the same latitude along columns.
        assert(np.sum(array[0, :, 1:] == array[1, :, 0:-1]) ==
               array[0, :, 1:].size)
        # 2 and 3 corners have the same latitude along columns.
        assert(np.sum(array[3, :, 1:] == array[2, :, 0:-1]) ==
               array[3, :, 1:].size)
        # Wraps around in the longitude direction.
        n_array = (array + 360) % 360
        assert(np.sum(n_array[0, :, 0] == n_array[1, :, -1] ) ==
               n_array[0, :, 0].size)
        assert(np.sum(n_array[3, :, 0] == n_array[2, :, -1] ) ==
               n_array[3, :, 0].size)


    input_clons = normalise_lons(input_clons)
    check_array(input_clons)
    check_array(input_clats)

    shape = (input_clons.shape[1] + 1, input_clons.shape[2])

    output_clons = np.zeros(shape)
    output_clons[:] = np.NAN
    output_clons[:-1,:] = input_clons[0,:,:] 
    output_clons[-1,:] = input_clons[3,-1,:]
    assert(np.sum(output_clons) != np.NAN)
    # All the numbers in input should also be in output. 
    # Weird that the rounding is necessary...
    assert(set(list(np.around(input_clons.flatten(), decimals=10))) ==
           set(list(np.around(output_clons.flatten(), decimals=10))))

    output_clats = np.zeros(shape)
    output_clats[:] = np.NAN
    output_clats[:-1,:] = input_clats[0,:,:] 
    output_clats[-1,:] = input_clats[3,-1,:]
    assert(np.sum(output_clats) != np.NAN)
    assert(set(list(np.around(input_clats.flatten(), decimals=10))) ==
           set(list(np.around(output_clats.flatten(), decimals=10))))


    return output_clons, output_clats 


class OASISGrid:
    """
    This class creates 3 files: 
        - areas.nc grid cell areas for atmos (t, u, v) and ice. 
        - masks.nc land sea mask for atmos (t, u, v) and ice. 
        - grids.nc lat, lon, rotation angle and corners for atmos (t, u, v) and
          ice.
    """

    def __init__(self, atm, ice, output_dir):
        self.atm = atm
        self.ice = ice

        self.areas_filename = os.path.join(output_dir, 'areas.nc')
        self.masks_filename = os.path.join(output_dir, 'masks.nc')
        self.grids_filename = os.path.join(output_dir, 'grids.nc')

    def make_areas(self):
        """
        Make netcdf file areas.nc with cice.srf, um1t.srf, um1u.srf, um1v.srf
        """

        ice = self.ice
        atm = self.atm

        f = nc.Dataset(self.areas_filename, 'w')

        f.createDimension('nyi', ice.num_lat_points)
        f.createDimension('nxi', ice.num_lon_points)
        f.createDimension('nyat', atm.num_lat_points)
        f.createDimension('nxat', atm.num_lon_points)
        f.createDimension('nyau', atm.num_lat_points)
        f.createDimension('nxau', atm.num_lon_points)
        f.createDimension('nyav', atm.num_lat_points - 1)
        f.createDimension('nxav', atm.num_lon_points)

        cice_srf = f.createVariable('cice.srf', 'f8', dimensions=('nyi', 'nxi'))
        cice_srf.units = 'm^2'
        cice_srf.title = 'cice grid T-cell area.'
        um1t_srf = f.createVariable('um1t.srf', 'f8',
                                    dimensions=('nyat', 'nxat'))
        um1t_srf.units = 'm^2'
        um1t_srf.title = 'um1t grid area.'
        um1u_srf = f.createVariable('um1u.srf', 'f8',
                                    dimensions=('nyau', 'nxau'))
        um1u_srf.units = 'm^2'
        um1u_srf.title = 'um1u grid area.'
        um1v_srf = f.createVariable('um1v.srf', 'f8',
                                    dimensions=('nyav', 'nxav'))
        um1v_srf.units = 'm^2'
        um1v_srf.title = 'um1v grid area.'

        cice_srf[:] = self.ice.area_t[:]
        um1t_srf[:] = atm.area_t[:]
        um1u_srf[:] = atm.area_u[:]
        um1v_srf[:] = atm.area_v[:]

        f.close()


    def make_masks(self):

        f = nc.Dataset(self.masks_filename, 'w')

        f.createDimension('ny0', self.ice.num_lat_points)
        f.createDimension('nx0', self.ice.num_lon_points)
        f.createDimension('ny1', self.atm.num_lat_points)
        f.createDimension('nx1', self.atm.num_lon_points)
        f.createDimension('ny2', self.atm.num_lat_points - 1)
        f.createDimension('nx2', self.atm.num_lon_points)

        # Make the ice mask.
        mask = f.createVariable('cice.msk', 'int32', dimensions=('ny0', 'nx0'))
        mask.units = '0/1:o/l'
        mask.title = 'Ice grid T-cell land-sea mask.'
        # Flip the mask. OASIS uses 1 = masked, 0 = unmasked.
        mask[:] = (1 - self.ice.mask[:]) 

        # Atm t mask.
        mask_t = f.createVariable('um1t.msk', 'int32', dimensions=('ny1', 'nx1'))
        mask_t.units = '0/1:o/l'
        mask_t.title = 'Atm grid T-cell land-sea mask.'
        # Build the mask using the atm land fraction. 
        mask_t[:] = np.copy(self.atm.landfrac)
        mask_t[np.where(self.atm.landfrac[:] != 1)] = 0

        # Atm u mask.
        mask_u = f.createVariable('um1u.msk', 'int32', dimensions=('ny1', 'nx1'))
        mask_u.units = '0/1:o/l'
        mask_u.title = 'Atm grid U-cell land-sea mask.'
        mask_u[:] = mask_t[:]
        # Hack? make u mask by adding land points onto Western land bounary.
        for i in range(mask_u.shape[0]):
            for j in range(mask_u.shape[1] - 1):
                if mask_u[i, j+1] == 1:
                    mask_u[i, j] = 1

        # Atm v mask.
        mask_v = f.createVariable('um1v.msk', 'int32', dimensions=('ny2', 'nx2'))
        mask_v.units = '0/1:o/l'
        mask_v.title = 'Atm grid V-cell land-sea mask.'
        mask_v[:] = mask_t[:-1,:]
        # Hack? make v mask by adding land points onto Southern land bounary.
        for i in range(mask_v.shape[0] - 1):
            for j in range(mask_v.shape[1]):
                if mask_v[i+1, j] == 1:
                    mask_v[i, j] = 1

        f.close()

    def make_grids(self):
        """
        lat, lon, and corners for atmos (t, u, v) and ice.
        """

        f = nc.Dataset(self.grids_filename, 'w')

        # Ice 
        f.createDimension('nyi', self.ice.num_lat_points)
        f.createDimension('nxi', self.ice.num_lon_points)
        f.createDimension('nci', 4)
        cice_lat = f.createVariable('cice.lat', 'f8', ('nyi', 'nxi'))
        cice_lat.units = "degrees_north"
        cice_lat.title = "cice grid T-cell latitude."
        cice_lat[:] = self.ice.y_t[:]

        cice_lon = f.createVariable('cice.lon', 'f8', ('nyi', 'nxi'))
        cice_lon.units = "degrees_east"
        cice_lon.title = "cice grid T-cell longitude."
        cice_lon[:] = self.ice.x_t[:]

        cice_cla = f.createVariable('cice.cla', 'f8', ('nci', 'nyi', 'nxi'))
        cice_cla.units = "degrees_north"
        cice_cla.title = "cice grid T-cell corner latitude"
        cice_cla[:] = self.ice.clat[:]

        cice_clo = f.createVariable('cice.clo', 'f8', ('nci', 'nyi', 'nxi'))
        cice_clo.units = "degrees_east"
        cice_clo.title = "cice grid T-cell corner longitude"
        cice_clo[:] = self.ice.clon[:]

        # Atm 
        f.createDimension('nya1', self.atm.num_lat_points)
        f.createDimension('nxa1', self.atm.num_lon_points)
        f.createDimension('nya2', self.atm.num_lat_points - 1)
        f.createDimension('nxa2', self.atm.num_lon_points)
        f.createDimension('nca', 4)

        # T cells. 
        um1t_lat = f.createVariable('um1t.lat', 'f8', ('nya1', 'nxa1'))
        um1t_lat.units = "degrees_north"
        um1t_lat.title = "um1t grid center latitude"
        um1t_lat[:] = self.atm.y_t[:]

        um1t_lon = f.createVariable('um1t.lon', 'f8', ('nya1', 'nxa1'))
        um1t_lon.units = "degrees_east"
        um1t_lon.title = "um1t grid center longitude"
        um1t_lon[:] = self.atm.x_t[:]

        um1t_clat = f.createVariable('um1t.cla', 'f8', ('nca', 'nya1', 'nxa1'))
        um1t_clat.units = "degrees_north"
        um1t_clat.title = "um1t grid corner latitude"
        um1t_clat[:] = self.atm.clat_t[:]

        um1t_clon = f.createVariable('um1t.clo', 'f8', ('nca', 'nya1', 'nxa1'))
        um1t_clon.units = "degrees_east"
        um1t_clon.title = "um1t grid corner longitude"
        um1t_clon[:] = self.atm.clon_t[:]

        # U cells
        um1u_lat = f.createVariable('um1u.lat', 'f8', ('nya1', 'nxa1'))
        um1u_lat.units = "degrees_north"
        um1u_lat.title = "um1u grid center latitude"
        um1u_lat[:] = self.atm.y_u[:]

        um1u_lon = f.createVariable('um1u.lon', 'f8', ('nya1', 'nxa1'))
        um1u_lon.units = "degrees_east"
        um1u_lon.title = "um1u grid center longitude"
        um1u_lon[:] = self.atm.x_u[:]

        um1u_clat = f.createVariable('um1u.cla', 'f8', ('nca', 'nya1', 'nxa1'))
        um1u_clat.units = "degrees_north"
        um1u_clat.title = "um1u grid corner latitude"
        um1u_clat[:] = self.atm.clat_u[:]

        um1u_clon = f.createVariable('um1u.clo', 'f8', ('nca', 'nya1', 'nxa1'))
        um1u_clon.units = "degrees_east"
        um1u_clon.title = "um1u grid corner longitude"
        um1u_clon[:] = self.atm.clon_u[:]

        # V cells.
        um1v_lat = f.createVariable('um1v.lat', 'f8', ('nya2', 'nxa2'))
        um1v_lat.units = "degrees_north"
        um1v_lat.title = "um1v grid center latitude"
        um1v_lat[:] = self.atm.y_v[:]

        um1v_lon = f.createVariable('um1v.lon', 'f8', ('nya2', 'nxa2'))
        um1v_lon.units = "degrees_east"
        um1v_lon.title = "um1v grid center longitude"
        um1v_lon[:] = self.atm.x_v[:]

        um1v_clat = f.createVariable('um1v.cla', 'f8', ('nca', 'nya2', 'nxa2'))
        um1v_clat.units = "degrees_north"
        um1v_clat.title = "um1v grid corner latitude"
        um1v_clat[:] = self.atm.clat_v[:]

        um1v_clon = f.createVariable('um1v.clo', 'f8', ('nca', 'nya2', 'nxa2'))
        um1v_clon.units = "degrees_east"
        um1v_clon.title = "um1v grid corner longitude"
        um1v_clon[:] = self.atm.clon_v[:]

        f.close()


    def write(self):
        
        self.make_areas()
        self.make_masks()
        self.make_grids()


class UMGrid:
    """
    Interpolate the ocean mask onto the UM grid. 

    Creates the land fraction and mask, later used as an input to the UM. 
    """

    def __init__(self, um_restart, num_lon_points, num_lat_points, mom_grid,
                 output_dir):

        self.SOUTHERN_EXTENT = -89.995002746582031
        self.SOUTHERN_EXTENT_CNR = -89.999496459960938
        self.NORTHERN_EXTENT = 89.995002746582031
        self.NORTHERN_EXTENT_CNR = 89.999496459960938

        self.mom_grid = mom_grid
        self.um_restart = um_restart

        self.um_restart_output = os.path.join(output_dir, 
                                              os.path.basename(um_restart))
        self.lfrac_filename_nc = os.path.join(output_dir, 'lfrac.nc')
        self.lfrac_filename_um = os.path.join(output_dir, 'lfrac')
        self.mask_filename_nc = os.path.join(output_dir, 'qrparm.mask.nc')
        self.mask_filename_um = os.path.join(output_dir, 'qrparm.mask')

        self.num_lon_points = num_lon_points
        self.num_lat_points = num_lat_points
        self.corners = 4

        # Set lats and lons. 
        self.lon = np.linspace(0, 360, num_lon_points, endpoint=False)
        self.lat = np.linspace(-90, 90, num_lat_points)
        dx_half = 360.0 / num_lon_points / 2.0
        dy_half = (180.0 / (num_lat_points - 1) / 2.0)

        # Similar to lon, lat but specify the coordinate at every grid
        # point. Also it wraps along longitude.
        self.x_t = np.tile(self.lon, (num_lat_points, 1))
        self.y_t = np.tile(self.lat, (num_lon_points, 1))
        self.y_t = self.y_t.transpose()

        self.x_u = self.x_t + dx_half
        self.y_u = self.y_t
        self.x_v = self.x_t
        self.y_v = self.y_t + dy_half

        def make_corners(x, y, dx, dy):

            # Set grid corners, we do these one corner at a time. Start at the 
            # bottom left and go anti-clockwise. This is the OASIS convention.
            clon = np.empty((self.corners, x.shape[0], x.shape[1]))
            clon[:] = np.NAN
            clon[0,:,:] = x - dx
            clon[1,:,:] = x + dx
            clon[2,:,:] = x + dx
            clon[3,:,:] = x - dx
            assert(not np.isnan(np.sum(clon)))

            clat = np.empty((self.corners, x.shape[0], x.shape[1]))
            clat[:] = np.NAN
            clat[0,:,:] = y[:,:] - dy
            clat[1,:,:] = y[:,:] - dy
            clat[2,:,:] = y[:,:] + dy
            clat[3,:,:] = y[:,:] + dy

            # The bottom latitude band should always be Southern extent, for
            # all t, u, v. 
            clat[0, 0, :] = -90
            clat[1, 0, :] = -90

            # The top latitude band should always be Northern extent, for all
            # t, u, v. 
            clat[2, -1, :] = 90
            clat[3, -1, :] = 90

            assert(not np.isnan(np.sum(clat)))

            return clon, clat

        self.clon_t, self.clat_t = make_corners(self.x_t, self.y_t, dx_half, dy_half)
        self.clon_u, self.clat_u = make_corners(self.x_u, self.y_u, dx_half, dy_half)
        self.clon_v, self.clat_v = make_corners(self.x_v, self.y_v, dx_half, dy_half)

        # The Northerly v points are going to be beyond the domain. Remove these. 
        self.y_v = self.y_v[:-1,:]
        self.x_v = self.x_v[:-1,:]
        self.clat_v = self.clat_v[:,:-1,:]
        self.clon_v = self.clon_v[:,:-1,:]
        #self.area_v = self.area_v[:-1,:]

        # Now that the grid is made we fix it up. We don't go from -90 to 90
        # but from self.SOUTHERN_EXTENT to self.NORTHERN_EXTENT. As far as I
        # can tell this is due to the SCRIP remapping library not handling the
        # poles properly and making bad weights. There is a test for this in
        # tests/test_scrip_remapping.py. If the tests don't pass there's no
        # point running the model with those remapping files. 
        def fix_grid():
            self.lat[0] = self.SOUTHERN_EXTENT
            self.lat[-1] = self.NORTHERN_EXTENT
            self.y_t[0, :] = self.SOUTHERN_EXTENT
            self.y_t[-1, :] = self.NORTHERN_EXTENT
            self.y_u[0, :] = self.SOUTHERN_EXTENT
            self.y_u[-1, :] = self.NORTHERN_EXTENT

            def fix_corners(clat):
                clat[0, 0, :] = self.SOUTHERN_EXTENT_CNR
                clat[1, 0, :] = self.SOUTHERN_EXTENT_CNR
                clat[2, -1, :] = self.NORTHERN_EXTENT_CNR
                clat[3, -1, :] = self.NORTHERN_EXTENT_CNR

            fix_corners(self.clat_t)
            fix_corners(self.clat_u)
            fix_corners(self.clat_v)

        fix_grid()

        # Use corners to calculate areas.
        self.area_t = self.calc_area(self.clon_t, self.clat_t)
        self.area_u = self.calc_area(self.clon_u, self.clat_u)
        self.area_v = self.calc_area(self.clon_v, self.clat_v)

        # This is defined after a call to make_landfrac.
        self.landfrac = None
        self.mask_t = None
        self.mask_u = None
        self.mask_v = None


    def calc_area(self, clons, clats):
        """
        Calculate the area of lat-lon polygons. 

        We project sphere onto a flat surface using an equal area projection
        and then calculate the area of flat polygon.
        """

        def area_polygon(p):
            """
            Calculate the area of a polygon. 

            Input is a polygon represented as a list of (x,y) vertex
            coordinates, implicitly wrapping around from the last vertex to the
            first.

            See http://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon
            """

            def segments(v):
                return zip(v, v[1:] + [v[0]])

            return 0.5 * abs(sum(x0*y1 - x1*y0 
                                 for ((x0, y0), (x1, y1)) in segments(p)))


        areas = np.zeros_like(clons[0])
        areas[:] = np.NAN

        m = Basemap(projection='laea', resolution='h',
                    llcrnrlon=0, llcrnrlat=-90.0,
                    urcrnrlon=360, urcrnrlat=90.0, lat_0=-90, lon_0=0)

        x, y = m(clons, clats)

        for i in range(x[0, :].shape[0]):
            for j in range(x[0, :].shape[1]):
                areas[i, j] = area_polygon(zip(x[:, i, j], y[:, i, j]))

        assert(np.sum(areas) is not np.NAN)
        assert(np.min(areas) > 0)
        assert(abs(1 - np.sum(areas) / EARTH_AREA) < 2e-4) 
       
        return areas


    def make_antarctic_mask(self, southern_lat, grid_lats):
        """
        Create mask on grid_lats to mask out everything South of a particular
        lat. 
        """

        def find_nearest_larger(val, array):
            """
            Find the value which is nearest and larger than val in array.
            """

            s_array = np.sort(array, axis=None)
            r = np.searchsorted(s_array, val, side='right') 
            return s_array[r]

        mask = np.zeros_like(grid_lats, dtype=bool)

        # Find Southern latitude of the destination that matches source.
        closest = find_nearest_larger(southern_lat, grid_lats)
        excluded_row = np.where(closest == grid_lats)[0][0]

        # Expect that lower latitudes have lower indices. 
        assert(all(grid_lats[excluded_row] > grid_lats[excluded_row - 1]))
        # Mask out all latitude bands equal to and less than closest. 
        mask[0:excluded_row,:] = True

        return mask


    def make_landfrac(self):
        """
        Regrid the ocean mask to create new land-sea fraction. 
        """

        src_clons, src_clats = oasis_to_2d_corners(self.mom_grid.clon,
                                                     self.mom_grid.clat)
        dest_clons, dest_clats = oasis_to_2d_corners(self.clon_t, self.clat_t)

        # The source grid is not defined South of -81. The easiest way to 
        # deal with this is to mask out the destination during regridding
        # and then set it all to land. 
        ant_mask = self.make_antarctic_mask(np.min(self.mom_grid.y_t), self.y_t)

        # Set up regridder with source and destination grid defs. All lons are
        # normalised -180, 180
        src_lons = normalise_lons(self.mom_grid.x_t)
        dest_lons = normalise_lons(self.x_t)

        r = Regridder(src_lons, self.mom_grid.y_t, src_clons, src_clats, None, 
                      dest_lons, self.y_t, dest_clons, dest_clats, ant_mask)

        # Do regridding of mom ocean mask. This will result in an
        # 'ocean fraction' not a land fraction.
        self.landfrac = r.regrid(self.mom_grid.mask)

        # Check regridding, ensure that src and dest masses are close. 
        src_mass = np.sum(self.mom_grid.area_t * self.mom_grid.mask)
        dest_mass = np.sum(self.area_t * self.landfrac)
        #assert(np.isclose(1, src_mass / dest_mass, atol=1e-5))
        # FIXME: this is not very close!
        assert(np.isclose(1, src_mass / dest_mass, atol=1e-3))

        # The destination has been masked out over Antarctica for regridding 
        # purposes, set that area to land. 
        self.landfrac[np.where(ant_mask)] = 0

        # Flip so that we have land fraction, rather than ocean fraction. 
        self.landfrac[:] = abs(1 - self.landfrac[:])
        # Clean up points which have a very small land fraction. 
        self.landfrac[np.where(self.landfrac[:] < 0.01)] = 0
        self.landfrac[np.where(self.landfrac[:] > 1)] = 1


    def put_basic_header(self, file):
        """
        Put in the basic netcdf header elements: lat, lon, time.  
        """
        
        file.createDimension('longitude', self.num_lon_points)
        file.createDimension('latitude', self.num_lat_points)
        file.createDimension('t')

        lon = file.createVariable('longitude', 'f8', dimensions=('longitude'))
        lon.long_name = 'longitude'
        lon.standard_name = 'longitude'
        lon.units = 'degrees_east'
        lon.point_spacing = 'even'
        lon.module = ''

        lat = file.createVariable('latitude', 'f8', dimensions=('latitude'))
        lat.long_name = 'latitude'
        lat.standard_name = 'latitude'
        lat.units = 'degrees_north'
        lat.point_spacing = 'even'

        t = file.createVariable('t', 'f8', dimensions=('t'))
        t.long_name = 't'
        t.units = 'days since 0001-01-01 00:00:00'
        t.time_origin = '01-JAN-0001:00:00:00'
        t[0] = 0

        lon[:] = self.lon
        lat[:] = self.lat


    def write_landfrac(self, convert_to_um=False):
        """
        Write out the land fraction.  
        """

        assert(self.landfrac is not None)

        f = nc.Dataset(self.lfrac_filename_nc, 'w', format='NETCDF3_CLASSIC')
        
        # Put in basic header elements lat, lon, time etc. 
        self.put_basic_header(f)
        f.createDimension('ht', 1)

        ht = f.createVariable('ht', 'f8', dimensions=('ht'))
        ht.long_name = 'Height'
        ht.units = 'm'
        ht.positive = 'up'
        lsm = f.createVariable('lsm', 'f8',
                               dimensions=('t', 'ht', 'latitude', 'longitude'))
        lsm.name = 'lsm'
        lsm.title = 'Stash code = 505'
        lsm.title = 'Land fraction in grid box'
        lsm.valid_min = 0.0
        lsm.valid_max = 1.0

        lsm[0, 0, :, :] = self.landfrac[:]
        f.close()

        # Convert to UM format.
        if convert_to_um:
            mkancil = Mkancil()
            ret = mkancil.convert_lfrac()
            assert(ret == 0)
            assert(os.path.exists(self.lfrac_filename_um))


    def write_mask(self, convert_to_um=False):
        """
        Write out mask used by the UM.
        
        This mask is used to differentiate between points that have some land
        fraction and those which have none at all.
        """
        assert(self.landfrac is not None)

        f = nc.Dataset(self.mask_filename_nc, 'w', format='NETCDF3_CLASSIC')
        # Put in basic header elements lat, lon, time etc. 
        self.put_basic_header(f)

        f.createDimension('surface', 1)

        surface = f.createVariable('surface', 'f8', dimensions=('surface'))
        surface.long_name = 'Surface'
        surface.units = 'level'
        surface.positive = 'up'

        lsm = f.createVariable('lsm', 'f8',
                               dimensions=('t', 'surface', 'latitude', 'longitude'))
        lsm.name = 'lsm'
        lsm.title = 'LAND MASK (No halo) (LAND=TRUE)'
        lsm.valid_min = 0.0
        lsm.valid_max = 1.0

        # Make the mask using the land fraction.
        mask = np.copy(self.landfrac)
        mask[np.where(self.landfrac[:] != 0)] = 1
        lsm[0, 0, :, :] = mask[:]
        f.close()            

        # Convert to UM format.
        if convert_to_um:
            mkancil = Mkancil()
            ret = mkancil.convert_mask()
            assert(ret == 0)
            assert(os.path.exists(self.mask_filename_um))


    def write(self):

        self.write_landfrac()
        self.write_mask()

        shutil.copyfile(self.um_restart, self.um_restart_output)

        # Update the um restart with new mask and landfrac. 
        with nc.Dataset(self.mask_filename_nc) as mask_f:
            mask = np.copy(mask_f.variables['lsm'][0, 0, :, :])
            # Flip because we use True to mean masked, UM uses True to mean
            # land.
            mask = abs(1 - mask)
            
            with nc.Dataset(self.lfrac_filename_nc) as lfrac_f:
                lfrac = lfrac_f.variables['lsm'][:]
                remask(self.um_restart_output, mask, lfrac)


class CICEGrid:
    """
    Make a new CICE grid and land sea mask using the MOM5 ocean grid definition
    and land sea mask.
    """

    def __init__(self, mom_grid, output_dir):
        """
        All variables are calculated here. 
        """

        self.output_dir = output_dir
        self.mom_grid = mom_grid
        self.mask_filename = os.path.join(self.output_dir, 'kmt.nc')
        self.grid_filename = os.path.join(self.output_dir, 'grid.nc')

        # Copy some stuff from MOM. FIXME: better/different way to do this.
        # Perhaps calculate here instead of in mom. Inherit from MOM? 
        self.area_t = mom_grid.area_t
        self.area_u = mom_grid.area_u

        self.x_u = mom_grid.x_u
        self.y_u = mom_grid.y_u
        self.x_t = mom_grid.x_t
        self.y_t = mom_grid.y_t

        self.clat = mom_grid.clat
        self.clon = mom_grid.clon

        self.num_lon_points = self.x_t.shape[1]
        self.num_lat_points = self.x_t.shape[0]

        # The ocean grid is double density. Containing T and U points.  The
        # cice grid is single density with separate variables for the T and
        # U points. 
        self.htn = mom_grid.dx[2::2, 0::2] + mom_grid.dx[2::2, 1::2]
        self.hte = mom_grid.dy[0::2, 2::2] + mom_grid.dy[1::2, 2::2]

        self.angle = mom_grid.angle_dx[2::2,0:-1:2]
        self.angleT = mom_grid.angle_dx[1::2,1::2]

        with nc.Dataset(self.mom_grid.mask_filename) as f:
            self.mask = np.copy(f.variables['mask'])


    def write_grid(self):
        """
        __init__() has done all the work of calculating fields. Here we make
        the netcdf file and copy over. 
        """

        f_ice = nc.Dataset(self.grid_filename, 'w')

        # Create some dimensions. 
        f_ice.createDimension('nx', self.num_lon_points)
        f_ice.createDimension('ny', self.num_lat_points)
        f_ice.createDimension('nc', 4)

        # Make all CICE grid variables. 
        ulat = f_ice.createVariable('ulat', 'f8', dimensions=('ny', 'nx'))
        ulat.units = "radians"
        ulat.title = "Latitude of U points"
        ulon = f_ice.createVariable('ulon', 'f8', dimensions=('ny', 'nx'))
        ulon.units = "radians"
        ulon.title = "Longitude of U points"
        tlat = f_ice.createVariable('tlat', 'f8', dimensions=('ny', 'nx'))
        tlat.units = "radians"
        tlat.title = "Latitude of T points"
        tlon = f_ice.createVariable('tlon', 'f8', dimensions=('ny', 'nx'))
        tlon.units = "radians"
        tlon.title = "Longitude of T points"
        htn = f_ice.createVariable('htn', 'f8', dimensions=('ny', 'nx'))
        htn.units = "cm"
        htn.title = "Width of T cells on North side."
        hte = f_ice.createVariable('hte', 'f8', dimensions=('ny', 'nx'))
        hte.units = "cm"
        hte.title = "Width of T cells on East side."
        angle = f_ice.createVariable('angle', 'f8', dimensions=('ny', 'nx'))
        angle.units = "radians"
        angle.title = "Rotation angle of U cells."
        angleT = f_ice.createVariable('angleT', 'f8', dimensions=('ny', 'nx'))
        angleT.units = "radians"
        angleT.title = "Rotation angle of T cells."
        area_t = f_ice.createVariable('tarea', 'f8', dimensions=('ny', 'nx'))
        area_t.units = "m^2"
        area_t.title = "Area of T cells."
        area_u = f_ice.createVariable('uarea', 'f8', dimensions=('ny', 'nx'))
        area_u.units = "m^2"
        area_u.title = "Area of U cells."

        area_t[:] = self.area_t[:]
        area_u[:] = self.area_u[:]

        # Now convert units: degrees -> radians. 
        tlat[:] = np.deg2rad(self.y_t)
        tlon[:] = np.deg2rad(self.x_t)
        ulat[:] = np.deg2rad(self.y_u)
        ulon[:] = np.deg2rad(self.x_u)

        # Convert from m to cm. 
        htn[:] = self.htn * 100.
        hte[:] = self.hte * 100.

        angle[:] = np.deg2rad(self.angle[:])
        angleT[:] = np.deg2rad(self.angleT[:])

        f_ice.close()


    def write_mask(self):

        input = self.mom_grid.mask_filename

        # ncrename will update the file history. 
        shutil.copyfile(input, self.mask_filename)
        with nc.Dataset(self.mask_filename, 'r+') as f:
            f.renameVariable('mask', 'kmt')

    def write(self):
        self.write_mask()
        self.write_grid()


class MOMGrid:
    """
    See src/mom5/ocean_core/ocean_grids.F90 and
    MOM4_guide.pdf for a description of the mosaic MOM5 grid.
    """

    def __init__(self, grid_filename, mask_filename, output_dir):
        """
        All the work gets done here. Other grids can then access any MOM field
        with <mom_object>.field
        """

        self.grid_filename = os.path.join(output_dir,
                                          os.path.basename(grid_filename))
        self.mask_filename = os.path.join(output_dir,
                                          os.path.basename(mask_filename))

        shutil.copyfile(mask_filename, self.mask_filename)
        shutil.copyfile(grid_filename, self.grid_filename)

        with nc.Dataset(mask_filename) as f:
            self.mask = np.copy(f.variables['mask'])

        with nc.Dataset(grid_filename) as f:
            self.dy = np.copy(f.variables['dy'])
            self.dx = np.copy(f.variables['dx'])
            self.angle_dx = np.copy(f.variables['angle_dx'])

            self.make_corners(f)
            self.calc_t_and_u_areas(f)

    def calc_t_and_u_areas(self, f):
        """
        Calculate (add up) areas of T and U cells using the ocean areas. 
        """

        area = np.copy(f.variables['area'])
        self.area_t = np.zeros((area.shape[0]/2, area.shape[1]/2))
        self.area_u = np.zeros((area.shape[0]/2, area.shape[1]/2))

        # Add up areas, going clockwise from bottom left.
        self.area_t = area[0::2, 0::2] + area[1::2, 0::2] + \
                      area[1::2, 1::2] + area[0::2, 1::2]

        # These need to wrap around the globe. Copy ocn_area and add an extra
        # column at the end.
        area_ext = np.append(area[:], area[:, 0:1], axis=1)
        self.area_u = area_ext[0::2, 1::2] + area_ext[1::2, 1::2] + \
                      area_ext[1::2, 2::2] + area_ext[0::2, 2::2]


    def make_corners(self, f):
        """
        The standard mom grid includes t-cell corners be specifying the u, v
        grid. Here we extract that and put it into the format expected by
        the regridder and OASIS.
        """

        x = np.copy(f.variables['x'])
        y = np.copy(f.variables['y'])

        self.clon = np.zeros((4, x.shape[0] / 2, x.shape[1] / 2))
        self.clon[:] = np.NAN
        self.clat = np.zeros((4, x.shape[0] / 2, x.shape[1] / 2))
        self.clat[:] = np.NAN

        # Corner lats. 0 is bottom left and then counter-clockwise. 
        # This is the OASIS convention. 
        self.clat[0,:,:] = y[0:-1:2,0:-1:2]
        self.clat[1,:,:] = y[0:-1:2,2::2]
        self.clat[2,:,:] = y[2::2,2::2]
        self.clat[3,:,:] = y[2::2,0:-1:2]

        # Corner lons.
        self.clon[0,:,:] = x[0:-1:2,0:-1:2]
        self.clon[1,:,:] = x[0:-1:2,2::2]
        self.clon[2,:,:] = x[2::2,2::2]
        self.clon[3,:,:] = x[2::2,0:-1:2]

        # Select points from double density grid. Southern most U points are
        # excluded. also the last (Eastern) U points, they are duplicates of
        # the first.
        self.x_t = x[1::2,1::2]
        self.y_t = y[1::2,1::2]
        self.x_u = x[2::2,0:-1:2]
        self.y_u = y[2::2,0:-1:2]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("ocean_grid", help="The input ocean grid.")
    parser.add_argument("ocean_mask", help="The input ocean mask.")
    parser.add_argument("um_restart", help="The input UM restart to remasked.")
    parser.add_argument("--atm_longitude_points", default=192, help="""
                        The number of longitude points in the atmosphere grid.
                        """)
    parser.add_argument("--atm_latitude_points", default=145, help="""
                        The number of longitude points in the atmosphere grid.
                        """)
    parser.add_argument("--output_dir", default="outputs",
                        help="The output directory.")
    parser.add_argument("--make_um_ancils", action='store_true', default=False,
                        help="Run ancil_top to regrid UM ancillaries.")

    args = parser.parse_args()

    mom_grid = MOMGrid(args.ocean_grid, args.ocean_mask, args.output_dir)
    cice_grid = CICEGrid(mom_grid, args.output_dir)

    #  FIXME: can get the lat, lon from the UM restart.
    atm_grid = UMGrid(args.um_restart, args.atm_longitude_points,
                      args.atm_latitude_points, mom_grid, args.output_dir)
    atm_grid.make_landfrac()
    oasis_grid = OASISGrid(atm_grid, cice_grid, args.output_dir)

    oasis_grid.write()
    cice_grid.write()
    atm_grid.write()

    if args.make_um_ancils:
        ancil = AncilTop(args.output, atm_grid.mask_filename_um)
        ret = ancil.make_ancils()
        assert(ret == 0)


if __name__ == "__main__":
    sys.exit(main())
