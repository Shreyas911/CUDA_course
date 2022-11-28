
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_80 -O3 --ptxas-options=-v --use_fast_math

all:			stream_legacy stream_per_thread multithread_legacy multithread_per_thread

stream_legacy:		stream_test.cu Makefile
			nvcc ${NVCCFLAGS} stream_test.cu -o stream_legacy

stream_per_thread:	stream_test.cu Makefile
			nvcc ${NVCCFLAGS} --default-stream per-thread stream_test.cu -o stream_per_thread

multithread_legacy:	multithread_test.cu Makefile
			nvcc ${NVCCFLAGS} -Xcompiler=-fopenmp \
				multithread_test.cu -o multithread_legacy

multithread_per_thread:	multithread_test.cu Makefile
			nvcc ${NVCCFLAGS} --default-stream per-thread -Xcompiler=-fopenmp \
				multithread_test.cu -o multithread_per_thread

clean:
			rm -f stream_legacy stream_per_thread multithread_legacy multithread_per_thread
