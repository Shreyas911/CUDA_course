/*

This is based on an example developed by Mark Harris for his NVIDIA blog:

http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

-- I have added some timing to it

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

    // set up 8 streams

    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    // loop over 8 streams

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // do a Memcpy and launch a dummy kernel on the default stream
        cudaMemcpy(d_data,h_data,sizeof(float),cudaMemcpyHostToDevice);
        kernel<<<1, 1>>>(d_data, 0);
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
