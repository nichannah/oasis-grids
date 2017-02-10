
import sys

sys.path.append('./esmgrids')
from esmgrids.mom_grid import MomGrid
from esmgrids.cice_grid import CiceGrid
from esmgrids.nemo_grid import NemoGrid
from esmgrids.t42_grid import T42Grid
from esmgrids.fv300_grid import FV300Grid
from esmgrids.core2_grid import Core2Grid
from esmgrids.jra55_grid import Jra55Grid

def factory(model_name, model_hgrid, model_mask, model_rows, model_cols):

    if model_name == 'MOM':
        model_grid = MomGrid.fromfile(model_hgrid,
                                      mask_file=model_mask)
    elif model_name == 'CICE':
        model_grid = CiceGrid.fromfile(model_hgrid,
                                       mask_file=model_mask)
    elif model_name == 'NEMO':
        model_grid = NemoGrid(model_hgrid, mask_file=model_mask)
    elif model_name == 'SPE':
        model_grid = T42Grid(model_cols, model_rows, 1, model_mask,
                                      description='Spectral')
    elif model_name == 'FVO':
        model_grid = FV300Grid(model_cols, model_rows, 1, model_mask,
                                          description='FV')
    elif model_name == 'CORE2':
        model_grid = Core2Grid()
    elif model_name == 'JRA55':
        model_grid = Jra55Grid()
    else:
        assert False

    return model_grid
