CUDA	:= $(CUDA_INSTALL_PATH)
INC	:= -I$(CUDA)/include -I.
LIB	:= -L$(CUDA)/lib64

NVCCFLAGS	:= -arch=sm_20 --ptxas-options=-v --use_fast_math \
		   -D BLOCK_X=$(BLOCK_X) -D BLOCK_Y=$(BLOCK_Y)
LIBS	:= -lcudart

laplace3d_$(ID):	laplace3d.cu laplace3d_kernel.cu
		nvcc laplace3d.cu -o bin/laplace3d_$(ID) \
		$(INC) $(LIB) $(NVCCFLAGS) $(LIBS)
