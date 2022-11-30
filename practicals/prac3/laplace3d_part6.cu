//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 16
#define BLOCK_Y 8

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <laplace3d_kernel.h>

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_laplace3d(int NX, int NY, int NZ, float* h_u1, float* h_u2);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  // 'h_' prefix - CPU (host) memory space

  int    NX=512, NY=511, NZ=511, REPEAT=10,
         bx, by, i, j, k, ind;
  float  *h_u1, *h_u2, *h_u3, *h_foo, err;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u1, *d_u2, *d_foo;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u3 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  checkCudaErrors( cudaMalloc((void **)&d_u1, sizeof(float)*NX*NY*NZ) );
  checkCudaErrors( cudaMalloc((void **)&d_u2, sizeof(float)*NX*NY*NZ) );

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

  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(d_u1, h_u1, sizeof(float)*NX*NY*NZ,
                              cudaMemcpyHostToDevice) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\nCopy u1 to device: %.1f (ms) \n", milli);

  // Set up the execution configuration

  bx = 1 + (NX-1)/BLOCK_X;
  by = 1 + (NY-1)/BLOCK_Y;

  dim3 dimGrid(bx,by);
  dim3 dimBlock(BLOCK_X,BLOCK_Y);

  // printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
  // printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

  // Execute GPU kernel

  cudaEventRecord(start);

  for (i = 1; i <= REPEAT; ++i) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    getLastCudaError("GPU_laplace3d execution failed\n");

    d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u2
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\n%dx GPU_laplace3d: %.1f (ms) \n", REPEAT, milli);

  // Read back GPU results

  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(h_u2, d_u1, sizeof(float)*NX*NY*NZ,
                              cudaMemcpyDeviceToHost) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\nCopy u2 to host: %.1f (ms) \n", milli);

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u2[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // Gold treatment

  cudaEventRecord(start);
  for (int i = 1; i <= REPEAT; ++i) {
    Gold_laplace3d(NX, NY, NZ, h_u1, h_u3);
    h_foo = h_u1; h_u1 = h_u3; h_u3 = h_foo;   // swap h_u1 and h_u3
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\n%dx Gold_laplace3d: %.1f (ms) \n \n", REPEAT, milli);

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u1[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // error check

  err = 0.0;

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;
        err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
      }
    }
  }

  printf("rms error = %f \n",sqrt(err/ (float)(NX*NY*NZ)));

 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_u1) );
  checkCudaErrors( cudaFree(d_u2) );
  free(h_u1);
  free(h_u2);
  free(h_u3);

  cudaDeviceReset();
}
