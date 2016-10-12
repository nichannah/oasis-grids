
from __future__ import print_function

import pytest
import os
import subprocess as sp
import sh
import numpy as np
import netCDF4 as nc

data_tarball = 'test_data.tar.gz'
data_tarball_url = 'http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz'

def check_vars_exist(areas, grids, masks):

    # Check that outputs and variables exist.
    assert(os.path.exists(areas))
    with nc.Dataset(areas) as f:
        assert(f.variables.has_key('momt.srf'))
        assert(f.variables.has_key('momu.srf'))

        assert(f.variables.has_key('nemt.srf'))
        assert(f.variables.has_key('nemu.srf'))
        assert(f.variables.has_key('nemv.srf'))

        assert(f.variables.has_key('t42t.srf'))

        assert(f.variables.has_key('fv3t.srf'))

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

        assert(f.variables.has_key('t42t.lat'))
        assert(f.variables.has_key('t42t.lon'))
        assert(f.variables.has_key('t42t.cla'))
        assert(f.variables.has_key('t42t.clo'))

        assert(f.variables.has_key('fv3t.lat'))
        assert(f.variables.has_key('fv3t.lon'))
        assert(f.variables.has_key('fv3t.cla'))
        assert(f.variables.has_key('fv3t.clo'))

    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        assert(f.variables.has_key('momt.msk'))
        assert(f.variables.has_key('momu.msk'))

        assert(f.variables.has_key('nemt.msk'))
        assert(f.variables.has_key('nemu.msk'))
        assert(f.variables.has_key('nemv.msk'))

        assert(f.variables.has_key('t42t.msk'))

        assert(f.variables.has_key('fv3t.msk'))


def check_var_values(areas, grids, masks):

    assert(os.path.exists(masks))
    with nc.Dataset(masks) as f:
        keys = ['momt.msk', 'momu.msk', 'nemt.msk', 'nemu.msk', 'nemv.msk', 't42t.msk', 'fv3t.msk']
        for k in keys:
            mask = f.variables[k][:]
            # Don't want it to be all masked.
            assert np.sum(mask) < mask.shape[0] * mask.shape[1]



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

        outputs = [output_areas, output_grids, output_masks]
        for f in [output_areas, output_grids, output_masks]:
            if os.path.exists(f):
                os.remove(f)

        my_dir = os.path.dirname(os.path.realpath(__file__))

        mom_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_mask.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + mom_args
        ret = sp.call(cmd)
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

        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + mom_args
        ret = sp.call(cmd)
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

    def test_all(self, input_dir, output_grids, output_areas, output_masks):

        outputs = [output_areas, output_grids, output_masks]
        for f in [output_areas, output_grids, output_masks]:
            if os.path.exists(f):
                os.remove(f)

        my_dir = os.path.dirname(os.path.realpath(__file__))

        mom_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_mask.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + mom_args
        ret = sp.call(cmd)
        assert(ret == 0)

        nemo_hgrid = os.path.join(input_dir, 'coordinates.nc')
        nemo_mask = os.path.join(input_dir, 'mesh_mask.nc')
        nemo_args = ['--model_hgrid', nemo_hgrid, '--model_mask', nemo_mask,
                     '--grids', output_grids, '--areas', output_areas,
                     '--masks', output_masks, 'NEMO']
        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + nemo_args
        ret = sp.call(cmd)
        assert(ret == 0)

        t42_mask = os.path.join(input_dir, 'lsm.20040101000000.nc')
        t42_args = ['--model_mask', t42_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'T42']
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + t42_args
        ret = sp.call(cmd)
        assert(ret == 0)

        fv300_mask = os.path.join(input_dir, 'lsm.20040101000000.nc')
        fv300_args = ['--model_mask', fv300_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'FV300']
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + fv300_args
        ret = sp.call(cmd)
        assert(ret == 0)

        check_vars_exist(output_areas, output_grids, output_masks)
        check_var_values(output_areas, output_grids, output_masks)

