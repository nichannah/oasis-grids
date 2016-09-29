
from __future__ import print_function

import pytest
import os
import subprocess as sp
import sh

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


    def test_mom(self, input_dir, output_grids, output_areas, output_masks):
        """
        Test script makes oasis grids from mom inputs.
        """

        outputs = [output_areas, output_grids, output_masks]
        for f in [output_areas, output_grids, output_masks]:
            if os.path.exists(f):
                os.remove(f)

        input_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        input_vgrid = os.path.join(input_dir, 'ocean_vgrid.nc')
        input_mask = os.path.join(input_dir, 'ocean_mask.nc')

        args = ['--model_hgrid', input_hgrid, '--model_vgrid', input_vgrid,
                '--model_mask', input_mask,
                '--grids', output_grids, '--areas', output_areas,
                '--masks', output_masks, 'MOM']

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + args
        ret = sp.call(cmd)
        assert(ret == 0)

        # Check that outputs exist.
        for f in outputs:
            assert(os.path.exists(f))

    def test_t42(self, input_dir, output_grids, output_areas, output_masks):

        outputs = [output_areas, output_grids, output_masks]
        for f in [output_areas, output_grids, output_masks]:
            if os.path.exists(f):
                os.remove(f)

        input_mask = os.path.join(input_dir, 'lsm.20040101000000.nc')

        args = ['--model_mask', input_mask,
                '--grids', output_grids, '--areas', output_areas,
                '--masks', output_masks, 'T42']

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')] + args
        ret = sp.call(cmd)
        assert(ret == 0)

        # Check that outputs exist.
        for f in outputs:
            assert(os.path.exists(f))

