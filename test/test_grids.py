
from __future__ import print_function

import pytest
import os
import subprocess as sp
import sh

data_tarball = 'test_data.tar.gz'
data_tarball_url = 'test_data.tar.gz'

class TestOasisGrids():

    @pytest.fixture
    def input_dir(self):
        if not os.path.exists('test_data'):
            if not os.path.exists(data_tarball):
                sh.wget(data_tarball)
            sh.tar('zxvf', data_tarball)

        return os.path.join(os.path.realpath('test_data'), 'input')

    @pytest.fixture
    def output_dir(self):
        return os.path.join(os.path.realpath('test_data'), 'output')

    def test_mom(self, input_dir, output_dir):
        """
        Test script makes oasis grids from mom inputs.
        """

        outputs = ['areas.nc', 'grids.nc', 'masks.nc']
        outputs = [os.path.join(output_dir, o) for o in outputs]
        for f in outputs:
            if os.path.exists(f):
                os.remove(f)

        input_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        input_hgrid = os.path.join(input_dir, 'ocean_hgrid.nc')
        input_mask = os.path.join(input_dir, 'ocean_mask.nc')

        args = [input_hgrid, input_vgrid, input_mask, '--output_dir',
                self.output_dir]

        cmd = [os.path.join(self.test_dir, '../', 'oasisgrids.py')] + args
        ret = sp.call(cmd)
        assert(ret == 0)

        # Check that outputs exist.
        for f in outputs:
            assert(os.path.exists(f))
