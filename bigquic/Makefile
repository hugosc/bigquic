CXX=g++
CXXFLAGS=-fopenmp -fPIC -pipe  -O3

all: bigquic-run

bigquic-run: bigquic-run.o bigquic.o
	$(CXX) -fopenmp -L/p/lib -L. -L../lib/ -o bigquic-run -fPIC -I/p/include -I../lib/metisLib bigquic-run.o bigquic.o -lmetis -lm -Wall

bigquic-run.o: bigquic-run.cpp bigquic.h
	$(CXX) -fopenmp -fPIC -I/p/include -I../lib/metisLib -c bigquic-run.cpp -Wall

bigquic.o: bigquic.h bigquic.cpp
	${CXX} ${CXXFLAGS} -I../lib/metisLib -L../lib/metisLib -c -o bigquic.o bigquic.cpp -Wall

clean:
	rm -rf *.o bigquic-run *.mexa64

