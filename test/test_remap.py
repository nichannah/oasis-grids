
import pytest
import sys, os
import subprocess as sp
import numpy as np
import numba
import netCDF4 as nc

sys.path.append('./esmgrids')
from esmgrids.mom_grid import MomGrid
from esmgrids.core2_grid import Core2Grid

from helpers import setup_test_input_dir, setup_test_output_dir
from helpers import calc_regridding_err

EARTH_RADIUS = 6370997.0
EARTH_AREA = 4*np.pi*EARTH_RADIUS**2

@numba.jit
def apply_weights(src, dest_shape, n_s, n_b, row, col, s):
    """
    Apply ESMF regirdding weights.
    """

    dest = np.ndarray(dest_shape).flatten()
    dest[:] = 0.0
    src = src.flatten()

    for i in range(n_s):
        dest[row[i]-1] = dest[row[i]-1] + s[i]*src[col[i]-1]

    return dest.reshape(dest_shape)


def remap(src_data, weights, src_grid, dest_grid):
    """
    Regrid a 2d field and see how it looks.
    """

    src_lats = src_grid.num_lat_points
    src_lons = src_grid.num_lon_points
    dest_lats = dest_grid.num_lat_points
    dest_lons = dest_grid.num_lon_points

    src_data = np.ndarray((src_lats, src_lons))
    dest_data = np.ndarray((dest_lats, dest_lons))

    for i in range(src_lats):
        src_data[i, :] = i

    with nc.Dataset(weights) as wf:
        n_s = wf.dimensions['n_s'].size
        n_b = wf.dimensions['n_b'].size
        row = wf.variables['row'][:]
        col = wf.variables['col'][:]
        s = wf.variables['S'][:]

    dest_data[:, :] = apply_weights(src_data[:, :], dest_data.shape,
                                       n_s, n_b, row, col, s)
    # compare src_data and dest_data
    if src_data.shape == dest_data.shape:
        np.allclose(dest_data, src_data)

    # Do a random sample: grab a couple of points at random on the
    # destination grid, find corrosponding point on the src grid and
    # compare.

    return dest_data

def remap_core2_to_mom(input_dir, output_dir, mom_hgrid, mom_mask):

    my_dir = os.path.dirname(os.path.realpath(__file__))
    cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

    core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')

    weights = os.path.join(output_dir, 'CORE2_MOM_conserve.nc')

    args = ['CORE2', 'MOM', '--src_grid', core2_hgrid,
            '--dest_grid', mom_hgrid, '--dest_mask', mom_mask,
            '--method', 'conserve', '--output', weights]
    ret = sp.call(cmd + args)
    assert ret == 0
    assert os.path.exists(weights)

    # Only use these to pull out the dimensions of the grids.
    mom = MomGrid.fromfile(mom_hgrid, mask_file=mom_mask)
    core2 = Core2Grid(core2_hgrid)

    src = np.empty_like(core2.x_t)
    for i in range(src.shape[0]):
        src[i, :] = i

    dest = remap(src, weights, core2, mom)
    return src, dest, weights


class TestRemap():

    @pytest.fixture
    def input_dir(self):
        return setup_test_input_dir()

    @pytest.fixture
    def output_dir(self):
        return setup_test_output_dir()

    @pytest.mark.accessom_tenth
    @pytest.mark.big_ram
    def test_core2_to_mom_tenth_weights(self, input_dir, output_dir):
        """
        Generate weights for core2 to MOM 0.1 remapping.
        """

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')
        core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')

        weights = os.path.join(output_dir, 'CORE2_MOM01_conserve.nc')

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]
        args = ['CORE2', 'MOM', '--src_grid', core2_hgrid,
                '--dest_grid', mom_hgrid, '--dest_mask', mom_mask,
                '--method', 'conserve', '--output', weights]
        ret = sp.call(cmd + args)
        assert ret == 0
        assert os.path.exists(weights)

    @pytest.mark.big_ram
    def test_core2_to_mom_tenth_remapping(self, input_dir):
        """
        Do a test remapping between core2 and MOM 0.1 grid. This is a superset
        of the test above.
        """

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')

        src, dest, weights = remap_core2_to_mom(input_dir, output_dir,
                                                mom_hgrid, mom_mask)

        rel_err = abs(src_tot - dest_tot) / dest_tot

        print('ESMF src_total {}'.format(src_tot))
        print('ESMF dest_total {}'.format(src_tot))
        print('ESMF relative error {}'.format(rel_err))

        assert np.allclose(src_tot, dest_tot, rtol=1e-15)


    def test_mom_to_mom_remapping(self, input_dir, output_dir):

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        output = os.path.join(output_dir, 'MOM_MOM_bilinear.nc')

        args = ['MOM', 'MOM', '--src_grid', mom_hgrid,
                '--dest_grid', mom_hgrid, '--output', output]
        ret = sp.call(cmd + args)
        assert ret == 0
        assert os.path.exists(output)

        # Only use these to pull out the dimensions of the grids.
        mom = MomGrid.fromfile(mom_hgrid, mask_file=mom_mask)

        src = np.empty_like(mom.x_t)
        for i in range(src.shape[0]):
            src[i, :] = i

        remap(src, output, mom, mom)

    @pytest.mark.conservation
    def test_core2_to_mom_one_remapping(self, input_dir, output_dir):

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        src, dest, weights = remap_core2_to_mom(input_dir, output_dir,
                                                mom_hgrid, mom_mask)

        # Write out remapped files.
        for name, data in [('esmf_src_field', src), ('esmf_dest_field', dest)]:
            with nc.Dataset(os.path.join(output_dir, name + '.nc'), 'w') as f:
                f.createDimension('ny', data.shape[0])
                f.createDimension('nx', data.shape[1])

                var = f.createVariable(name, 'f8', ('ny','nx'))
                var[:] = data[:]

        src_file = os.path.join(output_dir, 'esmf_src_field.nc')
        dest_file = os.path.join(output_dir, 'esmf_dest_field.nc')

        src_tot, dest_tot = calc_regridding_err(weights,
                                                src_file, 'esmf_src_field',
                                                dest_file, 'esmf_dest_field')
        rel_err = abs(src_tot - dest_tot) / dest_tot

        print('ESMF src_total {}'.format(src_tot))
        print('ESMF dest_total {}'.format(src_tot))
        print('ESMF relative error {}'.format(rel_err))

        assert np.allclose(src_tot, dest_tot, rtol=1e-15)


    @pytest.mark.areas
    def test_compare_areas(self, input_dir, output_dir):
        """
        Compare areas for CORE2 and MOM with those calculated by ESMF
        remapping.
        """

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        _, _, weights = remap_core2_to_mom(input_dir, output_dir,
                                           mom_hgrid, mom_mask)

        core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')
        core2 = Core2Grid(core2_hgrid)

        with nc.Dataset(weights) as f:
            area_t = f.variables['area_a'][:].reshape(core2.num_lat_points,
                                                      core2.num_lon_points)
            frac = f.variables['frac_a'][:].reshape(core2.num_lat_points,
                                                    core2.num_lon_points)
        area_t[:, :] = area_t[:, :]*EARTH_RADIUS**2

        assert np.allclose(core2.area_t,  area_t, rtol=1e-3)
        assert np.allclose(np.sum(core2.area_t), EARTH_AREA, rtol=1e-3)
        assert np.sum(area_t) == EARTH_AREA
