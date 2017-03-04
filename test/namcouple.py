
namcouple = \
"""\
 $NFIELDS
             1
 $END
 $NBMODEL
   2  {src_model} {dest_model}
 $END
 $RUNTIME
  21600
 $END
 $NLOGPRT
  1
 $END
 $STRINGS
src_field dest_field 1 7200  3 rst.nc EXPORTED
{src_x} {src_y} {dest_x} {dest_y} {src_grid} {dest_grid}
P  0  P  0
LOCTRANS MAPPING SCRIPR
INSTANT
rmp_{src_grid}_to_{dest_grid}_CONSERVE.nc src
CONSERV LR SCALAR LATLON 10 FRACNNEI FIRST
 $END
"""
