//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// template kernel routine
// 

template <int size>
__global__ void my_first_kernel(float *x)
{
  float xl[size];

  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  for (int i=0; i<size; i++) {
    xl[i] = expf((float) i*tid);
  }

  float sum = 0.0f;

  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      sum += xl[i]*xl[j];
    }
  }

  x[tid] = sum;
}


//
// CUDA routine to be called by main code
//

extern
int prac6(int nblocks, int nthreads)
{
  float *h_x, *d_x;
  int   nsize, n; 

  // allocate memory for arrays

  nsize = nblocks*nthreads ;

  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  // execute kernel for size=2

  my_first_kernel<2><<<nblocks,nthreads>>>(d_x);
  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %g \n",n,h_x[n]);

  // execute kernel for size=3

  my_first_kernel<3><<<nblocks,nthreads>>>(d_x);
  cudaMemcpy(h_x,d_x,nsize*sizeof(int),cudaMemcpyDeviceToHost);
  for (n=0; n<nsize; n++) printf(" n,  i  =  %d  %g \n",n,h_x[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

  return 0;
}

 
