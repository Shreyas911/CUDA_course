#include <stdio.h>
#include <cuda.h>

#include "helper_cuda.h"

__global__ void do_work(double *data, int N, int idx) {
	int i = blockIdx.x * blockDim.x + blockDim.x*idx + threadIdx.x;
	if (i < N) {
		for (int j = 0; j < 200; j++) {
			data[i] = cos(data[i]);
			data[i] = sqrt(fabs(data[i]));
		}
	}
}

int main()
{
	int nblocks = 30;
	int blocksize = 1024;
	double *data;
	checkCudaErrors(cudaMalloc( (void**)&data, nblocks*blocksize*sizeof(double) ));



	float time;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	dim3 dimBlock( blocksize, 1, 1 );
	dim3 dimGrid( 1, 1, 1 );
	for (int i = 0; i < nblocks; i++)
		do_work<<<dimGrid,dimBlock>>>(data, nblocks*blocksize, i);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("Serialised time:  %g ms\n", time);

	cudaStream_t streams[nblocks];
	for (int i = 0; i < nblocks; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));

	checkCudaErrors(cudaEventRecord(start, 0));
	checkCudaErrors(cudaEventSynchronize(start));
	for (int i = 0; i < nblocks; i++)
		do_work<<<dimGrid,dimBlock,0,streams[i]>>>(data, nblocks*blocksize, i);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("Multi-stream parallel time:  %g ms\n", time);

	for (int i = 0; i < nblocks; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));

	checkCudaErrors(cudaFree( data ));
	return EXIT_SUCCESS;
}

