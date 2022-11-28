
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_80 --ptxas-options=-v --use_fast_math -O3

all: kernel_overlap work_streaming

kernel_overlap:	kernel_overlap.cu Makefile
	nvcc kernel_overlap.cu -o kernel_overlap $(INC) $(NVCCFLAGS) $(LIB)

work_streaming:	work_streaming.cu Makefile
	nvcc work_streaming.cu -o work_streaming $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f kernel_overlap work_streaming

