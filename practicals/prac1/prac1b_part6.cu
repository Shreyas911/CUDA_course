//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x, float *v1, float *v2)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = v1[tid]+v2[tid];

  printf("Thread = %d, x = %f\n", tid, x[tid]);
}


//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x;
  float *h_v1, *d_v1, *h_v2, *d_v2;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));
  h_v1 = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_v1, nsize*sizeof(float)));
  h_v2 = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_v2, nsize*sizeof(float)));

  // Allocate on host

  for (n=0; n<nsize; n++){
	h_v1[n] = (float) n;
	h_v2[n] = (float) 2*n;
  }

  // Copy to device

checkCudaErrors( cudaMemcpy(d_v1,h_v1,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );
checkCudaErrors( cudaMemcpy(d_v2,h_v2,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );

  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>(d_x, d_v1, d_v2);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(d_x));
  free(h_x);
  checkCudaErrors(cudaFree(d_v1));
  free(h_v1);
  checkCudaErrors(cudaFree(d_v2));
  free(h_v2);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
