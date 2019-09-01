CPC = g++
CC = gcc
MPCC = mpicc
CPFLAGS = -Wall -std=c++11
CFLAGS = -Wall -std=c99 -fopenmp

objects = main.o algorithm.o managetrade.o io.o fullproblem.o subproblem.o kernels.o linked.o

gert: $(objects)
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -o gert $(objects) -lm

managetrade.o: managetrade.c managetrade.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

algorithm.o: algorithm.c algorithm.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

main.o: main.c svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

io.o: io.c io.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

subproblem.o: subproblem.c subproblem.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

fullproblem.o: fullproblem.c fullproblem.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

kernels.o: kernels.c kernels.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

linked.o: linked.c linked.h svm.h
	module load cports openmpi;\
	$(MPCC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f gert $(objects)

test: testPHP testStates testBasketball testPoly


testMP: gert
	module load cports openmpi;\
	mpiexec -n 2 ./gert -f testDataset.txt -s multiTrain.txt
