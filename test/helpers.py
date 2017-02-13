
import os
import sh

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

