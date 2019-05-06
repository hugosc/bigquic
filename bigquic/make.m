system('make bigquic.o');
mex -largeArrayDims -I../lib/metisLib/ CFLAGS="\$CFLAGS -fopenmp " LDFLAGS="\$LDFLAGS -fopenmp " COMPFLAGS="\$COMPFLAGS -openmp" -cxx bigquic.o ../lib/libmetis.a bigquic-mex.cpp
