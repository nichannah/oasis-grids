
import os
import sh
import netCDF4 as nc
import numpy as np

EARTH_RADIUS = 6370997.0

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, 'test_data')

def setup_test_input_dir():
    data_tarball_url = 'http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz'
    test_data_tarball = os.path.join(test_dir, 'test_data.tar.gz')

    if not os.path.exists(test_data_dir):
        if not os.path.exists(test_data_tarball):
            sh.wget('-P', test_dir, data_tarball_url)
        sh.tar('zxvf', test_data_tarball, '-C', test_dir)

    return os.path.join(test_data_dir, 'input')

def setup_test_output_dir():
    output_dir =  os.path.join(test_data_dir, 'output')
    for f in os.listdir(output_dir):
        p = os.path.join(f, output_dir)
        try:
            os.remove(p)
        except Exception as e:
            pass
    return output_dir

def calc_regridding_err(weights, src_file, src_field, dest_file, dest_field):
    """
    Calculate the regirdding error.
    """

    with nc.Dataset(src_file) as f:
        src = f.variables[src_field][:]
    with nc.Dataset(dest_file) as f:
        dest = f.variables[dest_field][:]

    with nc.Dataset(weights) as f:
        try:
            area_a = f.variables['area_a'][:]
        except KeyError as e:
            area_a = f.variables['src_grid_area'][:]
        area_a = area_a.reshape(src.shape[0], src.shape[1])
        area_a = area_a*EARTH_RADIUS**2

        try:
            area_b = f.variables['area_b'][:]
        except KeyError as e:
            area_b = f.variables['dst_grid_area'][:]
        area_b = area_b.reshape(dest.shape[0], dest.shape[1])
        area_b = area_b*EARTH_RADIUS**2

        try:
            frac_a = f.variables['frac_a'][:]
        except KeyError as e:
            frac_a = f.variables['src_grid_frac'][:]
        frac_a = frac_a.reshape(src.shape[0], src.shape[1])

        try:
            frac_b = f.variables['frac_b'][:]
        except KeyError as e:
            frac_b = f.variables['dst_grid_frac'][:]
        frac_b = frac_b.reshape(dest.shape[0], dest.shape[1])

    # Calculation of totals here.
    # http://www.earthsystemmodeling.org/esmf_releases/non_public/ESMF_5_3_0/ESMC_crefdoc/node3.html
    src_total = np.sum(src[:, :] * area_a[:, :] * frac_a[:, :])
    dest_total = np.sum(dest[:, :] * area_b[:, :])

    return src_total, dest_total
