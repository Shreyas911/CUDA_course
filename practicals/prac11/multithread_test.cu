/*

This is based on an example developed by Mark Harris for his NVIDIA blog:

http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

-- I have changed it into a multithreaded implementation with timing

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>

#include <cuda.h>

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    // initialise CUDA timing, and start timer

    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float  *h_data, *d_data;
    h_data = (float *) malloc(sizeof(float));
    cudaMalloc(&d_data, sizeof(float));
    h_data[0] = 1.0f;

    // set up 8 OpenMP threads

    const int num_threads = 8;
    omp_set_num_threads(num_threads);
    float *data[num_threads];

    // loop over num_threads

    for (int i = 0; i < num_threads; i++)
      cudaMalloc(&data[i], N * sizeof(float));

#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        printf(" thread ID = %d \n",omp_get_thread_num());
        
        // launch one worker kernel per thread
        kernel<<<1, 64>>>(data[i], N);
    }

    // wait for completion of all kernels

    cudaDeviceSynchronize();

    // stop timer and report execution time

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    printf("execution time (ms): %f \n",milli);

    cudaDeviceReset();

    return 0;
}
