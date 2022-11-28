
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -arch=sm_80 --ptxas-options=-v --use_fast_math

match:	match.cu match_gold.cpp
	nvcc match.cu match_gold.cpp -o match \
	$(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm match
