
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_80 --ptxas-options=-v --use_fast_math

scan:	scan.cu Makefile
	nvcc scan.cu -o scan $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm scan

