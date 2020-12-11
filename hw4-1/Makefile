NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
EXES     := hw4-1

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw4-1: hw4-1.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $? 
