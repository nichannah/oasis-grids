
import pytest
import os
import subprocess as sp
import sh
import numpy as np
import netCDF4 as nc

data_tarball = 'test_data.tar.gz'
data_tarball_url = 'http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz'

def cleanup(outputs):

    for f in outputs:
        if os.path.exists(f):
            os.remove(f)

def check_first_combo_vars_exist(areas, grids, masks):

    # Check that outputs and variables exist.
    assert(os.path.exists(areas))
    with nc.Dataset(areas) as f:
        assert(f.variables.has_key('momt.srf'))
        assert(f.variables.has_key('momu.srf'))

        assert(f.variables.has_key('mo1t.srf'))
        assert(f.variables.has_key('mo1u.srf'))

        assert(f.variables.has_key('nemt.srf'))
        assert(f.variables.has_key('nemu.srf'))
        assert(f.variables.has_key('nemv.srf'))

        assert(f.variables.has_key('spet.srf'))

        assert(f.variables.has_key('fvot.srf'))

    assert(os.path.exists(grids))
    with nc.Dataset(grids) as f:
        assert(f.variables.has_key('momt.lat'))
        assert(f.variables.has_key('momt.lon'))
        assert(f.variables.has_key('momt.cla'))
        assert(f.variables.has_key('momt.clo'))

        assert(f.variables.has_key('momu.lat'))
        assert(f.variables.has_key('momu.lon'))
        assert(f.variables.has_key('momu.cla'))
        assert(f.variables.has_key('momu.clo'))

        assert(f.variables.has_key('mo1t.lat'))
        assert(f.variables.has_key('mo1t.lon'))
        assert(f.variables.has_key('mo1t.cla'))
        assert(f.variables.has_key('mo1t.clo'))

        assert(f.variables.has_key('mo1u.lat'))
        assert(f.variables.has_key('mo1u.lon'))
        assert(f.variables.has_key('mo1u.cla'))
        assert(f.variables.has_key('mo1u.clo'))

        assert(f.variables.has_key('nemt.lat'))
        assert(f.variables.has_key('nemt.lon'))
        assert(f.variables.has_key('nemt.cla'))
        assert(f.variables.has_key('nemt.clo'))

        assert(f.variables.has_key('nemu.lat'))
        assert(f.variables.has_key('nemu.lon'))
        assert(f.variables.has_key('nemu.cla'))
        assert(f.variables.has_key('nemu.clo'))

        assert(f.variables.has_key('nemv.lat'))
        assert(f.variables.has_key('nemv.lon'))
        assert(f.variables.has_key('nemv.cla'))
        assert(f.variables.has_key('nemv.clo'))

        assert(f.variables.has_key('spet.lat'))
        assert(f.variables.has_key('spet.lon'))
        assert(f.variables.has_key('spet.cla'))
        assert(f.variables.has_key('spet.clo'))

        assert(f.variables.has_key('fvot.lat'))
        assert(f.variables.has_key('fvot.lon'))
        assert(f.variables.has_key('fvot.cla'))
        assert(f.variables.has_key('fvot.clo'))

    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        assert(f.variables.has_key('momt.msk'))
        assert(f.variables.has_key('momu.msk'))

        assert(f.variables.has_key('mo1t.msk'))
        assert(f.variables.has_key('mo1u.msk'))

        assert(f.variables.has_key('nemt.msk'))
        assert(f.variables.has_key('nemu.msk'))
        assert(f.variables.has_key('nemv.msk'))

        assert(f.variables.has_key('spet.msk'))

        assert(f.variables.has_key('fvot.msk'))


def check_accessom_tenth_vars_exist(output_areas, output_grids, output_masks)

    # Check that outputs and variables exist.
    assert(os.path.exists(areas))
    with nc.Dataset(areas) as f:
        assert(f.variables.has_key('momt.srf'))
        assert(f.variables.has_key('momu.srf'))

        assert(f.variables.has_key('cict.srf'))
        assert(f.variables.has_key('cicu.srf'))

        assert(f.variables.has_key('cort.srf'))

        assert(f.variables.has_key('jrat.srf'))

    assert(os.path.exists(grids))
    with nc.Dataset(grids) as f:
        assert(f.variables.has_key('momt.lat'))
        assert(f.variables.has_key('momt.lon'))
        assert(f.variables.has_key('momt.cla'))
        assert(f.variables.has_key('momt.clo'))

        assert(f.variables.has_key('momu.lat'))
        assert(f.variables.has_key('momu.lon'))
        assert(f.variables.has_key('momu.cla'))
        assert(f.variables.has_key('momu.clo'))

        assert(f.variables.has_key('cict.lat'))
        assert(f.variables.has_key('cict.lon'))
        assert(f.variables.has_key('cict.cla'))
        assert(f.variables.has_key('cict.clo'))

        assert(f.variables.has_key('cicu.lat'))
        assert(f.variables.has_key('cicu.lon'))
        assert(f.variables.has_key('cicu.cla'))
        assert(f.variables.has_key('cicu.clo'))

        assert(f.variables.has_key('cort.lat'))
        assert(f.variables.has_key('cort.lon'))
        assert(f.variables.has_key('cort.cla'))
        assert(f.variables.has_key('cort.clo'))

        assert(f.variables.has_key('jrat.lat'))
        assert(f.variables.has_key('jrat.lon'))
        assert(f.variables.has_key('jrat.cla'))
        assert(f.variables.has_key('jrat.clo'))

    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        assert(f.variables.has_key('momt.msk'))
        assert(f.variables.has_key('momu.msk'))

        assert(f.variables.has_key('cict.msk'))
        assert(f.variables.has_key('cicu.msk'))

        assert(f.variables.has_key('cort.msk'))

        assert(f.variables.has_key('jrat.msk'))


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


class TestOasisGrids():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = os.path.join(test_dir, 'test_data')
    test_data_tarball = os.path.join(test_dir, data_tarball)
    output_dir = os.path.join(test_data_dir, 'output')

    @pytest.fixture
    def input_dir(self):

        if not os.path.exists(self.test_data_dir):
            if not os.path.exists(self.test_data_tarball):
                sh.wget('-P', self.test_dir, data_tarball_url)
            sh.tar('zxvf', self.test_data_tarball, '-C', self.test_dir)

        return os.path.join(self.test_data_dir, 'input')

    @pytest.fixture
    def output_grids(self):
        return os.path.join(self.output_dir, 'grids.nc')

    @pytest.fixture
    def output_areas(self):
        return os.path.join(self.output_dir, 'areas.nc')

    @pytest.fixture
    def output_masks(self):
        return os.path.join(self.output_dir, 'masks.nc')

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
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + mom_args
        ret = sp.call(cmd + mom_args)
        assert(ret == 0)

        # MOM 1 deg grid
        mo1_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mo1_mask = os.path.join(input_dir, 'grid_spec.nc')
        mo1_args = ['--model_hgrid', mo1_hgrid, '--model_mask', mo1_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM1']
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


    def test_accessom_tenth(self, input_dir, output_grids, output_areas,
                            output_masks):
        """
        Test that a combination of MOM, NEMO, T42 spectral and FV grids can be
        combined together into the OASIS grids.
        """

        outputs = [output_areas, output_grids, output_masks]
        cleanup(outputs)

        # MOM 0.1 deg grid
        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + mom_args
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

        check_accessom_tenth_vars_exist(output_areas, output_grids, output_masks)
        keys = ['momt.msk', 'momu.msk', 'cict.msk', 'cicu.msk',
                'cort.msk', 'jrat.msk']
        check_masks_values(output_masks, keys)
        keys = ['momt.srf', 'momu.srf', 'cict.srf', 'cicu.srf',
                'cort.srf', 'jrat.srf']
        check_areas_values(output_areas, keys)

