
import pytest
import sys, os
import time
import subprocess as sp
import numpy as np
import numba
import netCDF4 as nc
from scipy import ndimage as nd

sys.path.append('./esmgrids')
from esmgrids.mom_grid import MomGrid
from esmgrids.core2_grid import Core2Grid
from esmgrids.jra55_grid import Jra55Grid

from helpers import setup_test_input_dir, setup_test_output_dir
from helpers import calc_regridding_err

EARTH_RADIUS = 6370997.0
EARTH_AREA = 4*np.pi*EARTH_RADIUS**2

def fill_mask_with_nearest_neighbour(field, field_mask):
    """
    This is the Python way using grid-box nearest neighbour, an alternative is
    to do nn based on geographic distance using the above.
    """

    new_data = np.ma.copy(field)

    ind = nd.distance_transform_edt(field_mask,
                                    return_distances=False,
                                    return_indices=True)
    new_data[:, :] = new_data[tuple(ind)]

    return new_data


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

    with nc.Dataset(weights) as wf:
        n_s = wf.dimensions['n_s'].size
        n_b = wf.dimensions['n_b'].size
        row = wf.variables['row'][:]
        col = wf.variables['col'][:]
        s = wf.variables['S'][:]

    dest_data[:, :] = apply_weights(src_data[:, :], dest_data.shape,
                                       n_s, n_b, row, col, s)

    return dest_data

def remap_atm_to_ocean(input_dir, output_dir, mom_hgrid, mom_mask, core2_or_jra='CORE2'):

    my_dir = os.path.dirname(os.path.realpath(__file__))
    cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

    if core2_or_jra == 'CORE2':
        atm_hgrid = os.path.join(input_dir, 't_10.0001.nc')
        atm_grid = Core2Grid(atm_hgrid)
    elif core2_or_jra == 'JRA55':
        atm_hgrid = os.path.join(input_dir, 't_10.1984.30Jun2016.nc')
        atm_grid = Jra55Grid(atm_hgrid)
    else:
        assert False

    weights = os.path.join(output_dir, 'ATM_MOM_conserve.nc')

    args = [core2_or_jra, 'MOM', '--src_grid', atm_hgrid,
            '--dest_grid', mom_hgrid, '--dest_mask', mom_mask,
            '--method', 'conserve', '--output', weights]
    ret = sp.call(cmd + args)
    assert ret == 0
    assert os.path.exists(weights)

    # Only use these to pull out the dimensions of the grids.
    mom = MomGrid.fromfile(mom_hgrid, mask_file=mom_mask)

    src = np.empty_like(atm_grid.x_t)
    for i in range(src.shape[0]):
        src[i, :] = i

    dest = remap(src, weights, (mom.num_lat_points, mom.num_lon_points))
    return src, dest, weights


def create_weights_mom_to_jra55(mom_hgrid, mom_mask, input_dir, output_dir):

    my_dir = os.path.dirname(os.path.realpath(__file__))
    cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

    jra55_hgrid = os.path.join(input_dir, 't_10.1984.30Jun2016.nc')

    jra55_to_cice = os.path.join(output_dir, 'rmp_jrat_to_cict_CONSERV.nc')
    args = ['JRA55', 'MOM', '--src_grid', jra55_hgrid,
            '--dest_grid', mom_hgrid,
            '--method', 'conserve', '--output', jra55_to_cice,
            '--output_convention', 'SCRIP']
    ret = sp.call(cmd + args)
    assert ret == 0
    assert os.path.exists(jra55_to_cice)

    jra55_to_cice_dist = os.path.join(output_dir, 'rmp_jrat_to_cict_DISTWGT.nc')
    args = ['JRA55', 'MOM', '--src_grid', jra55_hgrid,
            '--dest_grid', mom_hgrid,
            '--method', 'neareststod', '--output', jra55_to_cice_dist,
            '--output_convention', 'SCRIP']
    ret = sp.call(cmd + args)
    assert ret == 0
    assert os.path.exists(jra55_to_cice_dist)

    cice_to_jra55 = os.path.join(output_dir, 'rmp_cict_to_jrat_CONSERV.nc')
    args = ['MOM', 'JRA55', '--src_grid', mom_hgrid,
            '--src_mask', mom_mask, '--dest_grid', jra55_hgrid,
            '--method', 'conserve', '--ignore_unmapped',
            '--output', cice_to_jra55,
            '--output_convention', 'SCRIP']
    ret = sp.call(cmd + args)
    assert ret == 0
    assert os.path.exists(cice_to_jra55)


def remap_to_tenth(input_dir, output_dir, src_field, weights=None):
    """
    Remap MOM quarter degree to MOM tenth.

    This is used to remap OASIS and CICE restarts.
    """

    quarter_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
    quarter_mask = os.path.join(input_dir, 'ocean_mask.nc')
    mom_quarter = MomGrid.fromfile(quarter_hgrid, mask_file=quarter_mask)

    tenth_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
    tenth_mask = os.path.join(input_dir, 'ocean_01_mask.nc')
    mom_tenth = MomGrid.fromfile(tenth_hgrid)

    # The src_field has land points, fill these in to avoid masking mess.
    new_src_field = fill_mask_with_nearest_neighbour(src_field, mom_quarter.mask_t)

    if weights is None:
        my_dir = os.path.dirname(os.path.realpath(__file__))

        weights = os.path.join(output_dir, 'MOM025_MOM10th_conserve.nc')
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]
        args = ['MOM', 'MOM', '--src_grid', quarter_hgrid,
                '--dest_grid', tenth_hgrid, '--dest_mask', tenth_mask,
                '--method', 'conserve', '--output', weights]
        ret = sp.call(cmd + args)
        assert ret == 0

    assert os.path.exists(weights)

    dest_field = remap(new_src_field, weights,
                       (mom_tenth.num_lat_points, mom_tenth.num_lon_points))

    return dest_field, weights


def remap_core2_to_jra55(input_dir, output_dir, src_field, weights=None):
    """
    Remap core2 to jra55.

    This is used to remap OASIS restarts.
    """

    core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')
    jra55_hgrid = os.path.join(input_dir, 't_10.1984.30Jun2016.nc')
    jra55 = Jra55Grid(jra55_hgrid)

    if weights is None:
        my_dir = os.path.dirname(os.path.realpath(__file__))

        weights = os.path.join(output_dir, 'core2_jra55_conserve.nc')
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]
        args = ['CORE2', 'JRA55', '--src_grid', core2_hgrid,
                '--dest_grid', jra55_hgrid,
                '--method', 'conserve', '--output', weights]
        ret = sp.call(cmd + args)
        print('Made weights {}'.format(weights))
        assert ret == 0

    assert os.path.exists(weights)

    dest_field = remap(src_field, weights,
                       (jra55.num_lat_points, jra55.num_lon_points))

    return dest_field, weights


class TestRemap():

    @pytest.fixture
    def input_dir(self):
        return setup_test_input_dir()

    @pytest.fixture
    def output_dir(self):
        return setup_test_output_dir()

    @pytest.mark.accessom
    @pytest.mark.restarts_one
    def test_remap_restarts_one(self, input_dir, output_dir):
        """
        Remap restarts from core2 to jra55
        """

        fname = 'a2i.nc'
        jra55 = Jra55Grid(os.path.join(input_dir, 't_10.1984.30Jun2016.nc'))

        weights = None
        with nc.Dataset(os.path.join(output_dir, fname), 'w') as fd:
            with nc.Dataset(os.path.join(input_dir, fname), 'r') as fs:

                assert 'ny' in fs.dimensions
                assert 'nx' in fs.dimensions

                fd.createDimension('ny', jra55.num_lat_points)
                fd.createDimension('nx', jra55.num_lon_points)

                for vname in fs.variables:
                    if vname == 'time':
                        continue

                    vd = fd.createVariable(vname, 'f8', ('ny','nx'))

                    if len(fs.variables[vname].shape) == 3:
                        src = fs.variables[vname][0, :, :]
                    else:
                        src = fs.variables[vname][:, :]

                    vd[:], weights = remap_core2_to_jra55(input_dir,
                                            output_dir, src, weights)
                    print('remapped {}'.format(vname))

        assert os.path.exists(os.path.join(output_dir, fname))


    @pytest.mark.accessom
    @pytest.mark.big_ram
    @pytest.mark.restarts_tenth
    def test_remap_restarts_tenth(self, input_dir, output_dir):
        """
        Remap restarts from 0.25 to 0.1 degree.
        """

        # Input file are at a 1/4 degree.
        files = ['monthly_sstsss.nc', 'i2o.nc', 'i2a.nc', 'o2i.nc', 'u_star.nc']

        mom = MomGrid.fromfile(os.path.join(input_dir, 'ocean_01_hgrid.nc'))

        weights = None
        for fname in files:
            with nc.Dataset(os.path.join(output_dir, fname), 'w') as fd:
                with nc.Dataset(os.path.join(input_dir, fname), 'r') as fs:

                    assert 'ny' in fs.dimensions
                    assert 'nx' in fs.dimensions

                    fd.createDimension('ny', mom.num_lat_points)
                    fd.createDimension('nx', mom.num_lon_points)

                    for vname in fs.variables:
                        if vname == 'time':
                            continue

                        vd = fd.createVariable(vname, 'f8', ('ny','nx'))

                        if len(fs.variables[vname].shape) == 3:
                            src = fs.variables[vname][0, :, :]
                        else:
                            src = fs.variables[vname][:, :]

                        vd[:], weights = remap_to_tenth(input_dir, output_dir,
                                                        src, weights)

        for fname in files:
            assert os.path.exists(os.path.join(output_dir, fname))

    @pytest.mark.accessom
    @pytest.mark.big_ram
    def test_accessom_tenth_mom_core2_weights(self, input_dir, output_dir):
        """
        Create all weights needed for ACCESS-OM tenth. OASIS calls these:

        rmp_cict_to_cort_CONSERV_FRACNNEI.nc,
        rmp_cort_to_cict_CONSERV_FRACNNEI.nc,
        rmp_cort_to_cict_DISTWGT.nc
        """

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')
        core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')

        nt62_to_cice = os.path.join(output_dir, 'rmp_cort_to_cict_CONSERV.nc')
        args = ['CORE2', 'MOM', '--src_grid', core2_hgrid,
                '--dest_grid', mom_hgrid,
                '--method', 'conserve', '--output', nt62_to_cice,
                '--output_convention', 'SCRIP']
        ret = sp.call(cmd + args)
        assert ret == 0
        assert os.path.exists(nt62_to_cice)

        nt62_to_cice_dist = os.path.join(output_dir, 'rmp_cort_to_cict_DISTWGT.nc')
        args = ['CORE2', 'MOM', '--src_grid', core2_hgrid,
                '--dest_grid', mom_hgrid,
                '--method', 'neareststod', '--output', nt62_to_cice_dist,
                '--output_convention', 'SCRIP']
        ret = sp.call(cmd + args)
        assert ret == 0
        assert os.path.exists(nt62_to_cice_dist)

        cice_to_nt62 = os.path.join(output_dir, 'rmp_cict_to_cort_CONSERV.nc')
        args = ['MOM', 'CORE2', '--src_grid', mom_hgrid,
                '--src_mask', mom_mask, '--dest_grid', core2_hgrid,
                '--method', 'conserve', '--ignore_unmapped',
                '--output', cice_to_nt62,
                '--output_convention', 'SCRIP']
        ret = sp.call(cmd + args)
        assert ret == 0
        assert os.path.exists(cice_to_nt62)

    @pytest.mark.accessom
    @pytest.mark.big_ram
    @pytest.mark.jra55_tenth
    def test_accessom_tenth_mom_jra55_weights(self, input_dir, output_dir):
        """
        Create all weights needed for ACCESS-OM tenth. OASIS calls these:

        rmp_cict_to_jrat_CONSERV_FRACNNEI.nc,
        rmp_jrat_to_cict_CONSERV_FRACNNEI.nc,
        rmp_jrat_to_cict_DISTWGT.nc
        """

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')

        create_weights_mom_to_jra55(mom_hgrid, mom_mask, input_dir, output_dir)

    @pytest.mark.accessom
    @pytest.mark.big_ram
    @pytest.mark.jra55_quarter
    def test_accessom_qarter_mom_jra55_weights(self, input_dir, output_dir):
        """
        Create all weights needed for ACCESS-OM tenth. OASIS calls these:

        rmp_cict_to_jrat_CONSERV_FRACNNEI.nc,
        rmp_jrat_to_cict_CONSERV_FRACNNEI.nc,
        rmp_jrat_to_cict_DISTWGT.nc
        """

        mom_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_mask.nc')

        create_weights_mom_to_jra55(mom_hgrid, mom_mask, input_dir, output_dir)


    @pytest.mark.accessom
    @pytest.mark.jra55_one
    def test_accessom_one_mom_jra55_weights(self, input_dir, output_dir):
        """
        Create all weights needed for ACCESS-OM tenth. OASIS calls these:

        rmp_cict_to_jrat_CONSERV_FRACNNEI.nc,
        rmp_jrat_to_cict_CONSERV_FRACNNEI.nc,
        rmp_jrat_to_cict_DISTWGT.nc
        """

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        create_weights_mom_to_jra55(mom_hgrid, mom_mask, input_dir, output_dir)

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
        src, dest, weights = remap_atm_to_ocean(input_dir, output_dir,
                                                 mom_hgrid, mom_mask)
        t1 = time.time()
        rel_err = calc_regridding_err(weights, src, dest)

        print('ESMF relative error {}'.format(rel_err))
        print('ESMF time to make 0.1 degree weights and remap {}'.format(t1-t0))

        assert rel_err < 1e-13


    @pytest.mark.conservation
    def test_core2_to_mom_one_remapping(self, input_dir, output_dir):

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        t0 = time.time()
        src, dest, weights = remap_atm_to_ocean(input_dir, output_dir,
                                                mom_hgrid, mom_mask)
        t1 = time.time()
        rel_err = calc_regridding_err(weights, src, dest)

        print('ESMF relative error {}'.format(rel_err))
        print('ESMF time to make 1 degree weights and remap {}'.format(t1-t0))

        assert rel_err < 1e-15

    @pytest.mark.conservation
    @pytest.mark.jra55_one
    def test_jra55_to_mom_one_remapping(self, input_dir, output_dir):

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        t0 = time.time()
        src, dest, weights = remap_atm_to_ocean(input_dir, output_dir,
                                                 mom_hgrid, mom_mask,
                                                 core2_or_jra='JRA55')
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

        _, _, weights = remap_atm_to_ocean(input_dir, output_dir,
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
