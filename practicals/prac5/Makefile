
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcufft

NVCCFLAGS	:= -lineinfo -arch=sm_80 --ptxas-options=-v --use_fast_math

both:		simpleCUBLAS simpleCUFFT

simpleCUBLAS:	simpleCUBLAS.cpp Makefile
		g++ simpleCUBLAS.cpp -o simpleCUBLAS $(INC) $(LIB)

simpleCUFFT:	simpleCUFFT.cu Makefile
		nvcc simpleCUFFT.cu -o simpleCUFFT $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f simpleCUBLAS simpleCUFFT
