
from __future__ import print_function

import pytest
import os
import subprocess as sp
import sh
import netCDF4 as nc

data_tarball = 'test_data.tar.gz'
data_tarball_url = 'http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz'

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
        nemo_args = ['--model_hgrid', nemo_hgrid,
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

        # Check that outputs and variables exist.
        assert(os.path.exists(output_areas))
        with nc.Dataset(output_areas) as f:
            assert(f.variables.has_key('momt.srf'))
            assert(f.variables.has_key('momu.srf'))

            assert(f.variables.has_key('nemt.srf'))
            assert(f.variables.has_key('nemu.srf'))
            assert(f.variables.has_key('nemv.srf'))

            assert(f.variables.has_key('t42t.srf'))

        assert(os.path.exists(output_grids))
        with nc.Dataset(output_grids) as f:
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

        assert(os.path.exists(output_masks))
        with nc.Dataset(output_masks) as f:
            assert(f.variables.has_key('momt.msk'))
            assert(f.variables.has_key('momu.msk'))

            assert(f.variables.has_key('nemt.msk'))
            assert(f.variables.has_key('nemu.msk'))
            assert(f.variables.has_key('nemv.msk'))

            assert(f.variables.has_key('t42t.msk'))
