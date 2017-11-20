
import sys
import os

my_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(my_dir, './esmgrids'))
from esmgrids.mom_grid import MomGrid
from esmgrids.cice_grid import CiceGrid
from esmgrids.nemo_grid import NemoGrid
from esmgrids.t42_grid import T42Grid
from esmgrids.fv300_grid import FV300Grid
from esmgrids.core2_grid import Core2Grid
from esmgrids.jra55_grid import Jra55Grid
from esmgrids.jra55_river_grid import Jra55RiverGrid

def factory(model_name, model_hgrid, model_mask, model_rows=None, model_cols=None):

    if model_name == 'MOM':
        model_grid = MomGrid.fromfile(model_hgrid,
                                      mask_file=model_mask)
    elif model_name == 'CICE':
        model_grid = CiceGrid.fromfile(model_hgrid,
                                       mask_file=model_mask)
    elif model_name == 'NEMO':
        model_grid = NemoGrid(model_hgrid, mask_file=model_mask)
    elif model_name == 'SPE':
        if model_rows is None:
            model_rows = 64
        if model_cols is None:
            model_rows = 128
        model_grid = T42Grid(model_cols, model_rows, 1, model_mask,
                                      description='Spectral')
    elif model_name == 'FVO':
        if model_rows is None:
            model_rows = 64
        if model_cols is None:
            model_rows = 128
        model_grid = FV300Grid(model_cols, model_rows, 1, model_mask,
                                          description='FV')
    elif model_name == 'CORE2':
        model_grid = Core2Grid(model_hgrid)
    elif model_name == 'JRA55':
        model_grid = Jra55Grid(model_hgrid)
    elif model_name == 'JRA55_river':
        model_grid = Jra55RiverGrid(model_hgrid)
    else:
        assert False

    return model_grid
