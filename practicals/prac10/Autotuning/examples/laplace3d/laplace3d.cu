//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <laplace3d_kernel.cu>

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

  // 'h_' prefix - CPU (host) memory space

  int    NX=128, NY=128, NZ=128, REPEAT=5000,
         bx, by, i, j, k, ind, pitch;
  size_t pitch_bytes;
  float  *h_u1, *h_u2;

  double timer, elapsed;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u1, *d_u2, *d_foo;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card

  cutilDeviceInit(argc, argv);
 
  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  cutilSafeCall(cudaMallocPitch((void **)&d_u1, &pitch_bytes, sizeof(float)*NX, NY*NZ) );
  cutilSafeCall(cudaMallocPitch((void **)&d_u2, &pitch_bytes, sizeof(float)*NX, NY*NZ) );

  pitch = pitch_bytes/sizeof(float);

  // initialise u1
    
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  elapsed_time(&timer);
  cutilSafeCall(cudaMemcpy2D(d_u1, pitch_bytes,
                             h_u1, sizeof(float)*NX,
                             sizeof(float)*NX, NY*NZ,
                             cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaThreadSynchronize());
  elapsed = elapsed_time(&timer);
  printf("\nCopy u1 to device: %f (s) \n", elapsed);

  // Set up the execution configuration

  bx = 1 + (NX-1)/BLOCK_X;
  by = 1 + (NY-1)/BLOCK_Y;

  dim3 dimGrid(bx,by);
  dim3 dimBlock(BLOCK_X,BLOCK_Y);

  // printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
  // printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

  // Execute GPU kernel

  for (i = 1; i <= REPEAT; ++i) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, pitch, d_u1, d_u2);
    cutilCheckMsg("GPU_laplace3d execution failed\n");

    d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u2
  }

  cutilSafeCall(cudaThreadSynchronize());
  elapsed = elapsed_time(&timer);
  printf("\n%dx GPU_laplace3d: %f (s) \n", REPEAT, elapsed);

  // Read back GPU results

  cutilSafeCall(cudaMemcpy2D(h_u2, sizeof(float)*NX,
                             d_u1, pitch_bytes,
                             sizeof(float)*NX, NY*NZ,
                             cudaMemcpyDeviceToHost) );
  elapsed = elapsed_time(&timer);
  printf("\nCopy u2 to host: %f (s) \n", elapsed);

  // Release GPU and CPU memory

  cutilSafeCall(cudaFree(d_u1));
  cutilSafeCall(cudaFree(d_u2));
  free(h_u1);
  free(h_u2);

  cudaThreadExit();
}
