
import pytest
import sys, os
import subprocess as sp
import sh
import numpy as np
import numba
import netCDF4 as nc

sys.path.append('./esmgrids')
from esmgrids.mom_grid import MomGrid
from esmgrids.core2_grid import Core2Grid

data_tarball = 'test_data.tar.gz'
data_tarball_url = 'http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz'

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


def check_remapping_weights(weights, src_grid, dest_grid):
    """
    Regrid a 2d field and see how it looks.
    """

    src_lats = src_grid.num_lat_points
    src_lons = src_grid.num_lon_points
    dest_lats = dest_grid.num_lat_points
    dest_lons = dest_grid.num_lon_points

    src_data = np.ndarray((src_lats, src_lons))
    dest_data = np.ndarray((dest_lats, dest_lons))

    src_data[:, :] = np.arange(src_lats*src_lons).reshape(src_lats, src_lons)

    with nc.Dataset(weights) as wf:
        n_s = wf.dimensions['n_s'].size
        n_b = wf.dimensions['n_b'].size
        row = wf.variables['row'][:]
        col = wf.variables['col'][:]
        s = wf.variables['S'][:]

    dest_data[:, :] = apply_weights(src_data[:, :], dest_data.shape,
                                       n_s, n_b, row, col, s)
    # compare src_data and dest_data
    if src_data.shape == dest_data.shape:
        np.allclose(dest_data, src_data)

    # Do a random sample: grab a couple of points at random on the
    # destination grid, find corrosponding point on the src grid and
    # compare.


class TestRemap():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = os.path.join(test_dir, 'test_data')
    test_data_tarball = os.path.join(test_dir, data_tarball)
    out_dir = os.path.join(test_data_dir, 'output')

    @pytest.fixture
    def input_dir(self):

        if not os.path.exists(self.test_data_dir):
            if not os.path.exists(self.test_data_tarball):
                sh.wget('-P', self.test_dir, data_tarball_url)
            sh.tar('zxvf', self.test_data_tarball, '-C', self.test_dir)

        return os.path.join(self.test_data_dir, 'input')

    @pytest.fixture
    def output_dir(self):
        return self.out_dir

    @pytest.mark.slow
    def test_core2_to_mom_tenth_remapping(self, input_dir):

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

        mom_hgrid = os.path.join(input_dir, 'ocean_01_hgrid.nc')
        mom_mask = os.path.join(input_dir, 'ocean_01_mask.nc')

        args = ['MOM', 'CORE2', '--src_grid', mom_hgrid,
                '--src_mask', mom_mask] 
        ret = sp.call(cmd + args)
        assert(ret == 0)

    @pytest.mark.fast
    def test_mom_to_mom_remapping(self, input_dir, output_dir):

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        output = os.path.join(output_dir, 'MOM_MOM_bilinear.nc')

        args = ['MOM', 'MOM', '--src_grid', mom_hgrid,
                '--dest_grid', mom_hgrid, '--output', output]
        ret = sp.call(cmd + args)
        assert(ret == 0)
        assert os.path.exists(output)

        # Only use these to pull out the dimensions of the grids.
        mom = MomGrid.fromfile(mom_hgrid, mask_file=mom_mask)

        check_remapping_weights(output, mom, mom)

    @pytest.mark.fast
    def test_core2_to_mom_remapping(self, input_dir, output_dir):

        my_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [os.path.join(my_dir, '../', 'remapweights.py')]

        mom_hgrid = os.path.join(input_dir, 'grid_spec.nc')
        mom_mask = os.path.join(input_dir, 'grid_spec.nc')

        core2_hgrid = os.path.join(input_dir, 't_10.0001.nc')

        output = os.path.join(output_dir, 'CORE2_MOM_bilinear.nc')

        args = ['CORE2', 'MOM', '--src_grid', core2_hgrid,
                '--dest_grid', mom_hgrid, '--dest_mask', mom_mask, 
		'--output', output]
        ret = sp.call(cmd + args)
        assert(ret == 0)
        assert os.path.exists(output)

        # Only use these to pull out the dimensions of the grids.
        mom = MomGrid.fromfile(mom_hgrid, mask_file=mom_mask)
        core2 = Core2Grid(core2_hgrid)

        check_remapping_weights(output, core2, mom)

    def test_compare_to_low_res_oasis(self):
        """
        Write some basic Fortran code that remaps a single field using
        OASIS. Compare it to the approach above.
        """

        # Make oasis grids

        # Build models

        # Copy everything to a work directory

        # Run model, exchanging a single field

        pass
