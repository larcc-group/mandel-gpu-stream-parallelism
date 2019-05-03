
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
// #include <cuda_runtime.h>

#include <unistd.h>
#include <sys/time.h>

#include "cudabase.h"

__global__ void test(int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d: block %d, thread %d/%d\n", gid, blockIdx.x, threadIdx.x, blockDim.x-1);
    printf("%d: Dim (%d, %d), Block (%d, %d), thread (%d, %d)\n", gid, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(int argc, char const *argv[]){
    int N = 10;
    if (argc > 1) {
        N = atoi(argv[0]);
    }

    int devCount;
    CudaSafeCall( cudaGetDeviceCount(&devCount) );
    printf("Found %d CUDA-enabled devices:\n", devCount);
    for (int i = 0; i < devCount; ++i)
    {
        cudaSetDevice(i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("CUDA Device #%d: \"%s\" (%s)\n", i+1, devProp.name, devProp.integrated ? "integrated" : "dedicated");
        printf("    CUDA Driver / Runtime Version     %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("    Compute Capability:               %d.%d\n",  devProp.major, devProp.minor);
        printf("    Memory:\n");
        printf("      Total global memory:              %lu MB @ %d MHz\n",  devProp.totalGlobalMem/(1024*1024), devProp.memoryClockRate/1000);
        printf("      Total shared memory per block:    %lu KB\n",  devProp.sharedMemPerBlock/1024);
        printf("      Total shared memory per multip:   %lu KB\n",  devProp.sharedMemPerMultiprocessor/1024);
        printf("      Maximum memory pitch:             %lu MB\n",  devProp.memPitch/(1024*1024));
        printf("      Global memory bus width:          %d bits\n",  devProp.memoryBusWidth);
        printf("    Number of multiprocessors:        %d\n",  devProp.multiProcessorCount);
        printf("    Threads per Warp:                 %d\n",  devProp.warpSize);
        printf("    Max registers per block:          %d\n",  devProp.regsPerBlock);
        printf("    Max registers per multiprocessor: %d\n",  devProp.regsPerMultiprocessor);
        printf("    Max threads per multiprocessors:  %d\n",  devProp.maxThreadsPerMultiProcessor);
        printf("    Maximum threads per block:        %d\n",  devProp.maxThreadsPerBlock);
        printf("                            Dimension ( 0 x  1 x  2)\n");
        printf("    Threads per block dimension:      (%d x %d x %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("    Blocks per grid dimension:        (%d x %d x %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("    Clock rate:                       %d MHz\n",  devProp.clockRate/1000);
        printf("    Total constant memory:            %lu KB\n",  devProp.totalConstMem/1024);
        printf("    Texture alignment:                %lu\n",  devProp.textureAlignment);
        printf("    Features support:\n");
        printf("      Concurrent copy and execution:    %s, %d asynchronous engines\n",  (devProp.deviceOverlap ? "Yes" : "No"), devProp.asyncEngineCount);
        printf("      Kernel execution timeout:         %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        printf("      Device can map host memory:       %s\n",  (devProp.canMapHostMemory ? "Yes" : "No"));
        printf("      Unified address space:            %s\n",  (devProp.unifiedAddressing ? "Yes" : "No"));
        printf("      Cooperative kernel launch:        %s\n",  (devProp.cooperativeLaunch ? "Yes" : "No"));
        printf("      Concurrent kernel execution:      %s\n",  (devProp.concurrentKernels ? "Yes" : "No"));
        printf("      ECC support enabled:              %s\n",  (devProp.ECCEnabled ? "Yes" : "No"));
        printf("\n");
    }

	long start, end;
	struct timeval timecheck;
	gettimeofday(&timecheck, NULL);
	start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    int Nblocks = 2;
    dim3 Nthreads(2, 2);
    // int Nthreads = 4;
    int *ptr;
    CudaSafeCall( cudaMalloc((void**)&ptr, sizeof( int)) );

    printf("Starting kernel, %d blocks, (%d, %d) threads, n = %d\n", Nblocks, Nthreads.x, Nthreads.y, N);
    // printf("Starting kernel, %d blocks, %d threads, n = %d\n", Nblocks, Nthreads, N);
	test<<<Nblocks, Nthreads>>>(N);
    printf("Finished kernel\n");

    int i;
    CudaSafeCall( cudaMemcpy(&i, ptr, sizeof( int),cudaMemcpyDeviceToHost) );

	gettimeofday(&timecheck, NULL);
	end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
	printf("%s\t%ldms\n", argv[1], (end - start));

    CudaCheckError();
}

