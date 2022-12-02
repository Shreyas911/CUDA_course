

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void reduction_gold(float* odata, float* idata, const unsigned int len) 
{
  *odata = 0;
  for(int i=0; i<len; i++) *odata += idata[i];
  printf("CPU sum = %f\n", *odata); 
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_idata, float *d_global_sum)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int global_tid = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;

    // first, each thread loads data into shared memory
    temp[tid] = g_idata[global_tid];

    // next, we perform binary tree reduction

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory
    if (tid==0) atomicAdd(d_global_sum, temp[0]);
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, *reference, sum, *h_global_sum;;
  float *d_idata, *d_global_sum;

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks   = 2;
  num_elements = num_blocks*512;
  num_threads  = 512;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
  h_global_sum = (float*) malloc(sizeof(float));
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

  // compute reference solutions

  reference = (float*) malloc(mem_size);
  reduction_gold(&sum, h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_global_sum, sizeof(float)) );

 // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_idata, d_global_sum);
  cudaDeviceSynchronize();
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_global_sum, d_global_sum, sizeof(float),
                              cudaMemcpyDeviceToHost) );

  // check results

  printf("reduction error = %f\n",h_global_sum[0]-sum);

  // cleanup memory

  free(h_data);
  free(h_global_sum);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_global_sum) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
