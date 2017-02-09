
sys.path.append('./esmgrids')
from esmgrids.mom_grid import MomGrid
from esmgrids.cice_grid import CiceGrid
from esmgrids.nemo_grid import NemoGrid
from esmgrids.t42_grid import T42Grid
from esmgrids.fv300_grid import FV300Grid
from esmgrids.core2_grid import Core2Grid
from esmgrids.jra55_grid import Jra55Grid

def factory(model_name)

    if model_name == 'MOM':
        model_grid = MomGrid.fromfile(args.model_hgrid,
                                      mask_file=args.model_mask)
        cells = ('t', 'u')
    elif model_name == 'CICE':
        model_grid = CiceGrid.fromfile(args.model_hgrid,
                                       mask_file=args.model_mask)
        cells = ('t', 'u')
    elif model_name == 'NEMO':
        model_grid = NemoGrid(args.model_hgrid, mask_file=args.model_mask)
        cells = ('t', 'u', 'v')
    elif model_name == 'SPE':
        num_lons = args.model_cols
        num_lats = args.model_rows
        model_grid = T42Grid(num_lons, num_lats, 1, args.model_mask,
                                      description='Spectral')
        cells = ('t')
    elif model_name == 'FVO':
        num_lons = args.model_cols
        num_lats = args.model_rows
        model_grid = FV300Grid(num_lons, num_lats, 1, args.model_mask,
                                          description='FV')
        cells = ('t')
    elif model_name == 'CORE2':
        model_grid = Core2Grid()
        cells = ('t')
    elif model_name == 'JRA55':
        model_grid = Jra55Grid()
        cells = ('t')
    else:
        assert False

    return model_grid

