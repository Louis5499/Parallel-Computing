NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
LDFLAGS  := -lm
EXES     := hw4-2

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw4-2: hw4-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
