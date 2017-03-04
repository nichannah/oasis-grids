
F90=mpif90
FFLAGS=-Wall -g -O0 -I/usr/include -I./oasis3-mct/Linux/build/lib/psmile.MPI1/
LDFLAGS=-L./oasis3-mct/Linux/lib/ -L./ -lcoupler -lpsmile.MPI1 -lscrip -lnetcdff -lmct -lmpeu

all: atm.exe land.exe ice.exe ocean.exe

coupler.o: coupler.F90
	$(F90) -c $(FFLAGS) $< -o $@

libcoupler.a: coupler.o
	ar rcs libcoupler.a coupler.o

atm.exe: atm.F90 libcoupler.a
	mpif90 atm.F90 -o $@ $(LDFLAGS)

land.exe: land.F90 libcoupler.a
	mpif90 land.F90 -o $@ $(LDFLAGS)

ice.exe: ice.F90 libcoupler.a
	mpif90 ice.F90 -o $@ $(LDFLAGS)

ocean.exe: ocean.F90 libcoupler.a
	mpif90 ocean.F90 -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *.mod *.a *.exe
