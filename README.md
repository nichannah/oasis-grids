
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

```{shell}
```
