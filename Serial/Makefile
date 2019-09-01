CPC = g++
CC = gcc

CPFLAGS = -Wall -std=c++11
CFLAGS = -Wall -std=c99

objects = main.o io.o algorithm.o fullproblem.o subproblem.o kernels.o linked.o

gert: $(objects) svm.h Makefile
	$(CC) $(CFLAGS) -o gert $(objects) -lm

main.o: main.c svm.h Makefile
	$(CC) $(CFLAGS) -c $<

io.o: io.c io.h svm.h Makefile
	$(CC) $(CFLAGS) -c $<

subproblem.o: subproblem.c subproblem.h svm.h Makefile
	$(CC) $(CFLAGS) -c $<

fullproblem.o: fullproblem.c fullproblem.h svm.h Makefile
	$(CC) $(CFLAGS) -c $<

kernels.o: kernels.c kernels.h svm.h Makefile
	$(CC) $(CFLAGS) -c $<

linked.o: linked.c linked.h svm.h Makefile
	$(CC) $(CFLAGS) -c $<

algorithm.o: algorithm.c algorithm.h svm.h Makefile
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f gert $(objects)

test: testStates testPoly

testStates: gert
	./gert -f alt.txt -s altModel.txt
	./gert -f alt.txt -t altModel.txt

testPoly: gert
	./gert -f circleData.txt -k 1 -d 2 -g 1 -s circleModel.txt
	./gert -f circleData.txt -t circleModel.txt -k 1 -d 2 -g 1
