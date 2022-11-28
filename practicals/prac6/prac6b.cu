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

template  <class T>
__global__ void my_first_kernel(T *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = threadIdx.x;
}


//
// CUDA routine to be called by main code
//

extern
int prac6(int nblocks, int nthreads)
{
  float *h_x, *d_x;
  int   *h_i, *d_i;
  int   nsize, n; 

  // allocate memory for arrays

  nsize = nblocks*nthreads ;

  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  h_i = (int *)malloc(nsize*sizeof(int));
  cudaMalloc((void **)&d_i, nsize*sizeof(int));

  // execute kernel for float

  my_first_kernel<<<nblocks,nthreads>>>(d_x);
  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // execute kernel for ints

  my_first_kernel<<<nblocks,nthreads>>>(d_i);
  cudaMemcpy(h_i,d_i,nsize*sizeof(int),cudaMemcpyDeviceToHost);
  for (n=0; n<nsize; n++) printf(" n,  i  =  %d  %d \n",n,h_i[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);
  cudaFree(d_i);
  free(h_i);

  return 0;
}

 
