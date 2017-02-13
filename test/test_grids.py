
import pytest
import os
import subprocess as sp
import sh
import numpy as np
import netCDF4 as nc

from helpers import setup_test_input_dir, setup_test_output_dir

def cleanup(outputs):

    for f in outputs:
        if os.path.exists(f):
            os.remove(f)

def check_first_combo_vars_exist(areas, grids, masks):

    # Check that outputs and variables exist.
    assert(os.path.exists(areas))
    with nc.Dataset(areas) as f:
        assert('momt.srf' in f.variables)
        assert('momu.srf' in f.variables)

        assert('mo1t.srf' in f.variables)
        assert('mo1u.srf' in f.variables)

        assert('nemt.srf' in f.variables)
        assert('nemu.srf' in f.variables)
        assert('nemv.srf' in f.variables)

        assert('spet.srf' in f.variables)

        assert('fvot.srf' in f.variables)

    assert(os.path.exists(grids))
    with nc.Dataset(grids) as f:
        assert('momt.lat' in f.variables)
        assert('momt.lon' in f.variables)
        assert('momt.cla' in f.variables)
        assert('momt.clo' in f.variables)

        assert('momu.lat' in f.variables)
        assert('momu.lon' in f.variables)
        assert('momu.cla' in f.variables)
        assert('momu.clo' in f.variables)

        assert('mo1t.lat' in f.variables)
        assert('mo1t.lon' in f.variables)
        assert('mo1t.cla' in f.variables)
        assert('mo1t.clo' in f.variables)

        assert('mo1u.lat' in f.variables)
        assert('mo1u.lon' in f.variables)
        assert('mo1u.cla' in f.variables)
        assert('mo1u.clo' in f.variables)

        assert('nemt.lat' in f.variables)
        assert('nemt.lon' in f.variables)
        assert('nemt.cla' in f.variables)
        assert('nemt.clo' in f.variables)

        assert('nemu.lat' in f.variables)
        assert('nemu.lon' in f.variables)
        assert('nemu.cla' in f.variables)
        assert('nemu.clo' in f.variables)

        assert('nemv.lat' in f.variables)
        assert('nemv.lon' in f.variables)
        assert('nemv.cla' in f.variables)
        assert('nemv.clo' in f.variables)

        assert('spet.lat' in f.variables)
        assert('spet.lon' in f.variables)
        assert('spet.cla' in f.variables)
        assert('spet.clo' in f.variables)

        assert('fvot.lat' in f.variables)
        assert('fvot.lon' in f.variables)
        assert('fvot.cla' in f.variables)
        assert('fvot.clo' in f.variables)

    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        assert('momt.msk' in f.variables)
        assert('momu.msk' in f.variables)

        assert('mo1t.msk' in f.variables)
        assert('mo1u.msk' in f.variables)

        assert('nemt.msk' in f.variables)
        assert('nemu.msk' in f.variables)
        assert('nemv.msk' in f.variables)

        assert('spet.msk' in f.variables)

        assert('fvot.msk' in f.variables)


def check_accessom_tenth_vars_exist(areas, grids, masks):

    # Check that outputs and variables exist.
    assert(os.path.exists(areas))
    with nc.Dataset(areas) as f:
        assert('momt.srf' in f.variables)
        assert('momu.srf' in f.variables)

        assert('cict.srf' in f.variables)
        assert('cicu.srf' in f.variables)

        assert('cort.srf' in f.variables)

        assert('jrat.srf' in f.variables)

    assert(os.path.exists(grids))
    with nc.Dataset(grids) as f:
        assert('momt.lat' in f.variables)
        assert('momt.lon' in f.variables)
        assert('momt.cla' in f.variables)
        assert('momt.clo' in f.variables)

        assert('momu.lat' in f.variables)
        assert('momu.lon' in f.variables)
        assert('momu.cla' in f.variables)
        assert('momu.clo' in f.variables)

        assert('cict.lat' in f.variables)
        assert('cict.lon' in f.variables)
        assert('cict.cla' in f.variables)
        assert('cict.clo' in f.variables)

        assert('cicu.lat' in f.variables)
        assert('cicu.lon' in f.variables)
        assert('cicu.cla' in f.variables)
        assert('cicu.clo' in f.variables)

        assert('cort.lat' in f.variables)
        assert('cort.lon' in f.variables)
        assert('cort.cla' in f.variables)
        assert('cort.clo' in f.variables)

        assert('jrat.lat' in f.variables)
        assert('jrat.lon' in f.variables)
        assert('jrat.cla' in f.variables)
        assert('jrat.clo' in f.variables)

    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        assert('momt.msk' in f.variables)
        assert('momu.msk' in f.variables)

        assert('cict.msk' in f.variables)
        assert('cicu.msk' in f.variables)

        assert('cort.msk' in f.variables)

        assert('jrat.msk' in f.variables)


def check_masks_values(masks, keys):

    # Check that masks is the right way around.
    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        for k in keys:
            mask = f.variables[k][:]
            # Don't want it to be all masked.
            assert np.sum(mask) < mask.shape[0] * mask.shape[1]

def check_areas_values(areas, keys):

    # Check that the areas are roughly correct.
    EARTH_AREA = 510072000e6
    assert(os.path.exists(areas))
    with nc.Dataset(areas) as f:
        for k in keys:
            area = f.variables[k][:]
            assert abs(1 - np.sum(area) / EARTH_AREA) < 5e-2


def check_for_holes(grids):
    """
    For proper consevative remapping, the corners of a cell have to
    coincide with the corners of its neighbour cell, with no “holes”
    between the cells.
    """

    # FIXME: do this.
    pass


class TestOasisGrids():
    @pytest.fixture
    def input_dir(self):
        return setup_test_input_dir()

    @pytest.fixture
    def output_grids(self):
        output_dir = setup_test_output_dir()
        return os.path.join(output_dir, 'grids.nc')

    @pytest.fixture
    def output_areas(self):
        output_dir = setup_test_output_dir()
        return os.path.join(output_dir, 'areas.nc')

    @pytest.fixture
    def output_masks(self):
        output_dir = setup_test_output_dir()
        return os.path.join(output_dir, 'masks.nc')

    def test_double_write(self, input_dir, output_grids, output_areas, output_masks):
        """
        Test that oasis_grids.py can safely be called twice with the same arguments.
        """

        outputs = [output_areas, output_grids, output_masks]
        cleanup(outputs)

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')]

        mom_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_mask.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        ret = sp.call(cmd + mom_args)
        assert(ret == 0)

        with nc.Dataset(output_grids) as fg:
            lon = fg.variables['momt.lon'][:]
            lat = fg.variables['momt.lat'][:]
            cla = fg.variables['momt.cla'][:]
            clo = fg.variables['momt.clo'][:]

        with nc.Dataset(output_areas) as fa:
            tsrf = fa.variables['momt.srf'][:]
            usrf = fa.variables['momu.srf'][:]

        with nc.Dataset(output_masks) as fm:
            tmsk = fm.variables['momt.msk'][:]
            umsk = fm.variables['momu.msk'][:]

        ret = sp.call(cmd + mom_args)
        assert(ret == 0)

        with nc.Dataset(output_grids) as fg:
            assert np.array_equal(fg.variables['momt.lon'][:], lon[:])
            assert np.array_equal(fg.variables['momt.lat'][:], lat[:])
            assert np.array_equal(fg.variables['momt.clo'][:], clo[:])
            assert np.array_equal(fg.variables['momt.cla'][:], cla[:])

        with nc.Dataset(output_areas) as fa:
            assert np.array_equal(fa.variables['momt.srf'][:], tsrf[:])
            assert np.array_equal(fa.variables['momu.srf'][:], usrf[:])

        with nc.Dataset(output_masks) as fm:
            assert np.array_equal(fm.variables['momt.msk'][:], tmsk[:])
            assert np.array_equal(fm.variables['momu.msk'][:], umsk[:])


    def test_first_combo(self, input_dir, output_grids, output_areas,
                         output_masks):
        """
        Test that a combination of MOM, NEMO, T42 spectral and FV grids can be
        combined together into the OASIS grids.
        """

        outputs = [output_areas, output_grids, output_masks]
        cleanup(outputs)

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] 

        # MOM 0.25 deg grid
        mom_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_mask.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        ret = sp.call(cmd + mom_args)
        assert(ret == 0)

        # MOM 1 deg grid. We supply a grid name here to distinguish from
        # the above
        mo1_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mo1_mask = os.path.join(input_dir, 'grid_spec.nc')
        mo1_args = ['--model_hgrid', mo1_hgrid, '--model_mask', mo1_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, '--grid_name', 'mo1', 'MOM']
        ret = sp.call(cmd + mo1_args)
        assert(ret == 0)

        nemo_hgrid = os.path.join(input_dir, 'coordinates.nc')
        nemo_mask = os.path.join(input_dir, 'mesh_mask.nc')
        nemo_args = ['--model_hgrid', nemo_hgrid, '--model_mask', nemo_mask,
                     '--grids', output_grids, '--areas', output_areas,
                     '--masks', output_masks, 'NEMO']
        my_dir = os.path.dirname(os.path.realpath(__file__))
        ret = sp.call(cmd + nemo_args)
        assert(ret == 0)

        # Don't include mask
        spe_args = ['--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'SPE']
        ret = sp.call(cmd + spe_args)
        assert(ret == 0)

        fvo_mask = os.path.join(input_dir, 'lsm.20040101000000.nc')
        fvo_args = ['--model_mask', fvo_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'FVO']
        ret = sp.call(cmd + fvo_args)
        assert(ret == 0)

        check_first_combo_vars_exist(output_areas, output_grids, output_masks)
        keys = ['momt.msk', 'momu.msk', 'mo1t.msk', 'mo1u.msk', 'nemt.msk',
                'nemu.msk', 'nemv.msk', 'spet.msk', 'fvot.msk']
        check_masks_values(output_masks, keys)
        keys = ['momt.srf', 'momu.srf', 'mo1t.srf', 'mo1u.srf', 'nemt.srf',
                'nemu.srf', 'nemv.srf', 'spet.srf', 'fvot.srf']
        check_areas_values(output_areas, keys)

    @pytest.mark.slow
    def test_accessom_tenth(self, input_dir, output_grids, output_areas,
                            output_masks):
        """
        """

        outputs = [output_areas, output_grids, output_masks]
        cleanup(outputs)

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')]

        # MOM 0.1 deg grid
        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        ret = sp.call(cmd + mom_args)
        assert(ret == 0)

        # CICE 0.1 deg grid
        cice_grid = os.path.join(input_dir, 'cice_01_grid.nc')
        cice_mask = os.path.join(input_dir, 'cice_01_mask.nc')
        cice_args = ['--model_hgrid', cice_grid, '--model_mask', cice_mask,
                     '--grids', output_grids, '--areas', output_areas,
                     '--masks', output_masks, 'CICE']
        ret = sp.call(cmd + cice_args)
        assert(ret == 0)

        # CORE2 grid
        core2_args = ['--grids', output_grids, '--areas', output_areas,
                      '--masks', output_masks, 'CORE2']
        ret = sp.call(cmd + core2_args)
        assert(ret == 0)

        # JRA55 grid
        jra55_args = ['--grids', output_grids, '--areas', output_areas,
                      '--masks', output_masks, 'JRA55']
        ret = sp.call(cmd + jra55_args)
        assert(ret == 0)

        check_accessom_tenth_vars_exist(output_areas, output_grids,
                                        output_masks)
        keys = ['momt.msk', 'momu.msk', 'cict.msk', 'cicu.msk',
                'cort.msk', 'jrat.msk']
        check_masks_values(output_masks, keys)
        keys = ['momt.srf', 'momu.srf', 'cict.srf', 'cicu.srf',
                'cort.srf', 'jrat.srf']
        check_areas_values(output_areas, keys)
