
# oasis-grids

Create OASIS3-MCT model coupling grid configurations.

This tool is a grid translator. It takes model grid definitions as netCDF files, converts to an Object/Python representation, then translates these to a format understood by the OASIS3-MCT coupler.

Build status [![Build Status](https://travis-ci.org/nicjhan/oasis-grids.svg?branch=master)](https://travis-ci.org/nicjhan/oasis-grids)

# Description

Coupling models together using the OASIS3-MCT coupler involves defining the model grids in a particular format. OASIS then uses these when regridding coupling fields. This tool can be used to create the OASIS grid definition files, called grids.nc, areas.nc, masks.nc for a collection of coupled models.

At a high level the OASIS regridding setup looks like this:

1. Start with model grid definitions as NetCDF files.
2. Offline transformation of above to OASIS grids (grids.nc, areas.nc, masks.nc).
3. OASIS uses grids from above and it's namcouple configuration to create regridding weights files for each pair of coupled grids. This only needs to be done once, thereafter the regridding weights files are reused.
4. During runtime the regridding weights files are used to interpolate fields between model grids.

The tool takes care of step 2 in this process. ~~In addition it provides an option to also do step 3. The advantage of doing step 3 offline, rather than depending on OASIS to do it, is that a more sophisticated (and a lot faster) regridding weight generation tool can be used.~~

# Use

The tool adds one model grid definition to the files grids.nc, areas.nc, masks.nc each time it is invoked. As such it is necessary to invoke it mulitiple times, once for every model which participates in the coupling.

For example to couple MOM with T42 atmosphere:

```{shell}
$ cd test
$ wget http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz
$ tar zxvf test_data.tar.gz
$ cd test_data/input
$ ../../oasisgrids.py MOM --model_hgrid ocean_hgrid.nc  --model_mask ocean_mask.nc \
    --grids grids.nc --areas areas.nc --masks masks.nc
$ ../../oasisgrids.py T42 --model_mask lsm.20040101000000.nc \
    --grids grids.nc --areas areas.nc --masks masks.nc
```

This first invocation of oasisgrids adds the MOM grid specification to the OASIS files, the second adds the T42 grid specification.

In this case no grid id/name is being provided and it will default to: lowercase model name postfixed with t, u, or v. The grid name can be given explicitly with the --grid_name option. The grid name needs to match what has been used in the OASIS namcouple file.

If there are multiple model configurations then all model grid variables can be added to the same OASIS files. For example with a T42 atmosphere coupled to both MOM and NEMO in different configs:

```{shell}
$ cd test
$ wget http://s3-ap-southeast-2.amazonaws.com/dp-drop/oasis-grids/test/test_data.tar.gz
$ tar zxvf test_data.tar.gz
$ cd test_data/input
$ ../../oasisgrids.py MOM --model_hgrid ocean_hgrid.nc --model_mask ocean_mask.nc \
    --grids grids.nc --areas areas.nc --masks masks.nc
$ ../../oasisgrids.py NEMO --model_hgrid coordinates.nc --model_mask data_1m_potential_temperature_nomask.nc \
    --grids grids.nc --areas areas.nc --masks masks.nc
$ ../../oasisgrids.py T42 --model_mask lsm.20040101000000.nc \
    --grids grids.nc --areas areas.nc --masks masks.nc
```

