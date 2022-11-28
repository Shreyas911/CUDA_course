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
	checkCudaErrors(cudaMallocHost((void**)&h_data, total_data*sizeof(double)));
	checkCudaErrors(cudaMalloc( (void**)&d_data, total_data*sizeof(double) ));

	//Initialise host data
	srand(0);
	for (int i = 0; i < total_data; i++)
		h_data[i] = (double)rand()/(double)RAND_MAX;

	int batches=8;
	cudaStream_t streams[batches];
	for (int i = 0; i < batches; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));

	//Start timing	
	float time;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	//Figure out how many blocks are needed
	int blocksize = 256;
	int data_fraction = (total_data-1)/batches+1;
	int nblocks = (data_fraction-1)/blocksize+1;

	for (int i = 0; i < batches; i++) {
		//Make sure we have the right size for each chunk
		int upload_size = data_fraction;
		if (i==batches-1) upload_size = total_data - data_fraction*i;

		//Copy data to device
		checkCudaErrors(cudaMemcpyAsync(&d_data[i*data_fraction],&h_data[i*data_fraction],upload_size*sizeof(double),cudaMemcpyHostToDevice,streams[i]));

		//Launch kernel to process data
		do_work<<<nblocks,blocksize,0,streams[i]>>>(d_data, total_data, i*nblocks);

		//Copy data back from device
		checkCudaErrors(cudaMemcpyAsync(&h_data[i*data_fraction],&d_data[i*data_fraction],upload_size*sizeof(double),cudaMemcpyDeviceToHost,streams[i]));
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	printf("Total processing time:  %g ms\n", time);

	for (int i = 0; i < batches; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	checkCudaErrors(cudaFree( d_data ));
	checkCudaErrors(cudaFreeHost(h_data));
	return EXIT_SUCCESS;
}

