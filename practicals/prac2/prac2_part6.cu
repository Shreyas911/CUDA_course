
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ float a, b, c;
__constant__ int   N_thread;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////


__global__ void average_thread(float *d_samples, float *d_avg_thread)
{
  float sum_thread, temp; 
  int   ind;

  // move array pointers to correct position
  // Similar to version 1
  // ind = threadIdx.x + N_thread*blockIdx.x*blockDim.x;
  ind = N_thread*threadIdx.x + N_thread*blockIdx.x*blockDim.x;

  // thread average calculation
  sum_thread = 0.0;
  for (int n=0; n<N_thread; n++) {
    temp = d_samples[ind];
    sum_thread += a*temp*temp + b*temp + c;
    // Similar to version 1
    // ind += blockDim.x;      // shift pointer to next element
    ind += 1;
  }
  
  // put thread average value into device array
  d_avg_thread[threadIdx.x + blockIdx.x*blockDim.x] = sum_thread/N_thread;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int     N=9600000;
  int     num_threads, h_N_thread;
  float   h_a, h_b, h_c;
  float  *h_avg_thread, *d_avg_thread, *d_samples;
  double  average;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_N_thread = 100;
  num_threads = N/h_N_thread;
  h_avg_thread = (float *)malloc(sizeof(float)*num_threads);

  checkCudaErrors( cudaMalloc((void **)&d_avg_thread, sizeof(float)*num_threads) );
  checkCudaErrors( cudaMalloc((void **)&d_samples, sizeof(float)*N) );

  // define constants and transfer to GPU

  h_a = 1.0f;
  h_b = 2.0f;
  h_c = 5.0f;

  checkCudaErrors( cudaMemcpyToSymbol(N_thread,    &h_N_thread,    sizeof(h_N_thread)) );
  checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  cudaEventRecord(start);
  checkCudaErrors( curandGenerateNormal(gen, d_samples, N, 0.0f, 1.0f) );
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, N/(0.001*milli));

  // execute kernel and time it

  cudaEventRecord(start);
  average_thread<<<num_threads/128, 128>>>(d_samples, d_avg_thread);
  getLastCudaError("average_thread execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Kernel execution time (ms): %f \n",milli);
  printf("effective bandwidth GB/s: %f \n", N*4.0/(0.001*milli)/1e9);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_avg_thread, d_avg_thread, sizeof(float)*num_threads,
                   cudaMemcpyDeviceToHost) );

  // compute average

  average = 0.0;
  for (int i=0; i<num_threads; i++) {
    average += h_avg_thread[i];
  }
  average = average/num_threads;
  printf("\nAverage value = %13.8f \n", average);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_avg_thread);
  checkCudaErrors( cudaFree(d_avg_thread) );
  checkCudaErrors( cudaFree(d_samples) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
