import pytest
import sys, os
import time
import shlex
import glob
import numpy as np
import subprocess as sp

from helpers import setup_test_input_dir
from helpers import calc_regridding_err

class TestOasis():
    """
    Run a basic OASIS example to test the generated config files.
    """

    @pytest.fixture
    def input_dir(self):
        return setup_test_input_dir()

    @pytest.fixture
    def oasis_dir(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(test_dir, 'oasis')

    @pytest.fixture
    def oasis3mct_dir(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(test_dir, 'oasis', 'oasis3-mct')

    def test_build(self, oasis_dir, oasis3mct_dir):
        """
        Build example Fortran code.
        """

        # First build oasis3-mct library.
        ret = sp.call(['make', '-C', oasis3mct_dir, 'ubuntu'])
        assert ret == 0

        # Build Fortran test code.
        ret = sp.call(['make', '-C', oasis_dir, 'clean'])
        assert ret == 0
        ret = sp.call(['make', '-C', oasis_dir])
        assert ret == 0

    @pytest.mark.conservation
    def test_remap_one_deg(self, input_dir, oasis_dir):
        """
        Use OASIS for a one degree remapping.
        """

        # Delete all netcdf files in oasis dir this will include the OASIS
        # configuration.
        for f in glob.glob(oasis_dir + '/*.nc'):
            try:
                os.remove(f)
            except FileNotFoundError as e:
                pass

        # Make oasis grids
        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'oasisgrids.py')]

        output_grids = os.path.join(oasis_dir, 'grids.nc')
        output_areas = os.path.join(oasis_dir, 'areas.nc')
        output_masks = os.path.join(oasis_dir, 'masks.nc')

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')
        mom_args = ['--model_hgrid', mom_hgrid, '--model_mask', mom_mask,
                    '--grids', output_grids, '--areas', output_areas,
                    '--masks', output_masks, 'MOM']
        ret = sp.call(cmd + mom_args)
        assert ret == 0

        core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')
        core2_args = ['--model_hgrid', core2_hgrid,
                      '--grids', output_grids, '--areas', output_areas,
                      '--masks', output_masks, 'CORE2']
        ret = sp.call(cmd + core2_args)
        assert ret == 0

        # Build models
        self.test_build(oasis_dir)

        # Run model, exchanging a single field
        cur_dir = os.getcwd()
        os.chdir(oasis_dir)

        cmd = 'mpirun -np 1 model_src : -np 1 model_dest'
        t0 = time.time()
        ret = sp.call(shlex.split(cmd))
        t1 = time.time()
        assert ret == 0

        os.chdir(cur_dir)

        # Look at the output of the field.
        weights = os.path.join(oasis_dir,
                               'rmp_cort_to_momt_CONSERV_FRACNNEI.nc')
        src_file = os.path.join(oasis_dir, 'src_field.nc')
        dest_file = os.path.join(oasis_dir, 'dest_field.nc')

        src_tot, dest_tot = calc_regridding_err(weights, src_file,
                                                'Array', dest_file, 'Array')
        rel_err = abs(src_tot - dest_tot) / dest_tot

        print('OASIS src_total {}'.format(src_tot))
        print('OASIS dest_total {}'.format(dest_tot))
        print('OASIS relative error {}'.format(rel_err))
        print('OASIS time to make weights and remap one field {}'.format(t1-t0))

        assert np.allclose(src_tot, dest_tot, rtol=1e-9)

