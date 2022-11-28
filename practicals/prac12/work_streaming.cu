#include <stdio.h>
#include <cuda.h>

#include "helper_cuda.h"

__global__ void do_work(double *data, int N, int idx) {
	int i = blockIdx.x * blockDim.x + blockDim.x*idx + threadIdx.x;
	if (i < N) {
		for (int j = 0; j < 20; j++) {
			data[i] = cos(data[i]);
			data[i] = sqrt(fabs(data[i]));
		}
	}
}

int main()
{
	//Allocate 1 GB of data
	int total_data = 1<<27;
	double *d_data, *h_data;
	h_data = (double*)malloc(total_data*sizeof(double));
	checkCudaErrors(cudaMalloc( (void**)&d_data, total_data*sizeof(double) ));

	//Initialise host data
	srand(0);
	for (int i = 0; i < total_data; i++)
		h_data[i] = (double)rand()/(double)RAND_MAX;

	//Start timing	
	float time;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	//Copy data to device
	checkCudaErrors(cudaMemcpy(d_data,h_data,total_data*sizeof(double),cudaMemcpyHostToDevice));

	//Figure out how many blocks are needed
	int blocksize = 256;
	int nblocks = (total_data-1)/blocksize+1;

	//Launch kernel to process data
	do_work<<<nblocks,blocksize,0,0>>>(d_data, total_data, 0*nblocks);

	//Copy data back from device
	checkCudaErrors(cudaMemcpy(h_data,d_data,total_data*sizeof(double),cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("Total processing time:  %g ms\n", time);

	checkCudaErrors(cudaFree( d_data ));
	free(h_data);
	return EXIT_SUCCESS;
}

