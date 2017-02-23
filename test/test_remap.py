
import pytest
import sys, os
import time
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


def remap(src_data, weights, dest_shape):
    """
    Regrid a 2d field and see how it looks.
    """

    dest_data = np.ndarray(dest_shape)

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

    dest = remap(src, weights, (mom.num_lat_points, mom.num_lon_points))
    return src, dest, weights


def remap_mom_one_to_tenth(input_dir, output_dir, src_field):
    """
    Remap MOM one degree to MOM tenth.

    This is used to remap OASIS and CICE restarts.
    """

    my_dir = os.path.dirname(os.path.realpath(__file__))

    one_hgrid = os.path.join(input_dir, 'grid_spec.nc')
    one_mask = os.path.join(input_dir, 'grid_spec.nc')
    tenth_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
    tenth_mask = os.path.join(input_dir, 'ocean_01_mask.nc')

    weights = os.path.join(output_dir, 'MOM1_MOM10th_conserve.nc')

    cmd = [os.path.join(my_dir, '../', 'remapweights.py')]
    args = ['MOM', 'MOM', '--src_grid', one_hgrid,
            '--dest_grid', tenth_hgrid, '--dest_mask', tenth_mask,
            '--method', 'conserve', '--output', weights]
    ret = sp.call(cmd + args)
    assert ret == 0
    assert os.path.exists(weights)

    dest_field = remap(src_field, weights,
                       (mom.num_lat_points, mom.num_lon_points))

    return dest_field


class TestRemap():

    @pytest.fixture
    def input_dir(self):
        return setup_test_input_dir()

    @pytest.fixture
    def output_dir(self):
        return setup_test_output_dir()

    @pytest.mark.restarts
    def test_remap_restarts(self, input_dir, output_dir):

        files = ['i2a.nc', 'i2o.nc', 'o2i.nc', 'u_star.nc']

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom = MomGrid.fromfile(mom_hgrid)

        for fname in files:
            with nc.Dataset(os.path.join(output_dir, fname), 'w') as fd:
                with nc.Dataset(os.path.join(input_dir, fname), 'r') as fs:

                    assert 'ny' in fs.dimensions
                    assert 'nx' in fs.dimensions

                    fd.createDimension('ny', mom.num_lat_points)
                    fd.createDimension('nx', mom.num_lon_points)

                    for vname in fs.variables:
                        vd = fd.createVariable(vname, 'f8', ('ny','nx'))

                        vd[:] = remap_mom_one_to_tenth(input_dir, output_dir,
                                                       fs.variables[vname][:])

        for fname in files:
            assert os.path.exists(os.path.join(output_dir, fname))


    def test_identical_remapping(self, input_dir, output_dir):
        """
        Remap between two identical grids
        """

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        output = os.path.join(output_dir, 'MOM_MOM_conserve.nc')

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

        remap(src, output, src.shape)

    @pytest.mark.big_ram
    @pytest.mark.conservation
    def test_core2_to_mom_tenth_remapping(self, input_dir, output_dir):
        """
        Do a test remapping between core2 and MOM 0.1 grid. 
        """

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')

        t0 = time.time()
        src, dest, weights = remap_core2_to_mom(input_dir, output_dir,
                                                mom_hgrid, mom_mask)
        t1 = time.time()
        rel_err = calc_regridding_err(weights, src, dest)

        print('ESMF relative error {}'.format(rel_err))
        print('ESMF time to make 0.1 degree weights and remap {}'.format(t1-t0))

        assert rel_err < 1e-15


    @pytest.mark.conservation
    def test_core2_to_mom_one_remapping(self, input_dir, output_dir):

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        t0 = time.time()
        src, dest, weights = remap_core2_to_mom(input_dir, output_dir,
                                                mom_hgrid, mom_mask)
        t1 = time.time()
        rel_err = calc_regridding_err(weights, src, dest)

        print('ESMF relative error {}'.format(rel_err))
        print('ESMF time to make 1 degree weights and remap {}'.format(t1-t0))

        assert rel_err < 1e-15


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
