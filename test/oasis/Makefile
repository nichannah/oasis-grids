
F90=mpif90
FFLAGS=-Wall -g -O0 -I./oasis3-mct/Linux/build/lib/psmile.MPI1/
LDFLAGS=-L./oasis3-mct/Linux/lib/ -L./ -lcoupler -lpsmile.MPI1 -lscrip -lnetcdff -lmct -lmpeu

all: model_src model_dest

libcoupler.a: coupler.o
	ar rcs libcoupler.a coupler.o

coupler.o: coupler.F90
	$(F90) -c $(FFLAGS) $< -o $@

model_src: libcoupler.a
	mpif90 model_src.F90 -o $@ $(LDFLAGS)

model_dest: libcoupler.a
	mpif90 model_dest.F90 -o $@ $(LDFLAGS)
