
all: model_src model_dest

coupler:
	mpif90 -c error_handler.F90 coupler.F90 -I./oasis3-mct/Linux/build/lib/psmile.MPI1/ -L./oasis3-mct/Linux/lib/ -lpsmile.MPI1 -lscrip -lnetcdff -lmct -lmpeu

model_src:
	mpif90 model_src.F90 -o model_src

model_dest:
	gfortran model_dest.F90 -o model_dest
