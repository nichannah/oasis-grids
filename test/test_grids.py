
from __future__ import print_function

import pytest
import os
import subprocess as sp
import sh

data_tarball = 'test_data.tar.gz'
data_tarball_url = 'http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz'

class TestOasisGrids():

    @pytest.fixture
    def input_dir(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, 'test_data')
        test_data_tarball = os.path.join(test_dir, data_tarball)

        if not os.path.exists(test_data_dir):
            if not os.path.exists(test_data_tarball):
                sh.wget('-P', test_dir, data_tarball_url)
            sh.tar('zxvf', test_data_tarball, '-C', test_dir)

        return os.path.join(test_data_dir, 'input')

    @pytest.fixture
    def output_dir(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, 'test_data')

        return os.path.join(test_data_dir, 'output')

    def test_mom(self, input_dir, output_dir):
        """
        Test script makes oasis grids from mom inputs.
        """

        output_areas = os.path.join(output_dir, 'areas.nc')
        output_grids = os.path.join(output_dir, 'grids.nc')
        output_masks = os.path.join(output_dir, 'masks.nc')
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
