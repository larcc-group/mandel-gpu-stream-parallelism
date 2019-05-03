/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*

   Author: Marco Aldinucci.
   email:  aldinuc@di.unipi.it
   marco@pisa.quadrics.com
   date :  15/11/97

Modified by:

****************************************************************************
 *  Author: Dalvan Griebler <dalvangriebler@gmail.com>
 *  Author: Dinei Rockenbach <dinei.rockenbach@edu.pucrs.br>
 *
 *  Copyright: GNU General Public License
 *  Description: This program simply computes the mandelbroat set.
 *  File Name: mandel.cpp
 *  Version: 1.0 (25/05/2018)
 *  Compilation Command: make
 ****************************************************************************
*/

#include <stdio.h>
#if !defined(NO_DISPLAY)
#include "marX2.h"
#endif
#include <sys/time.h>
#include <math.h>

#include <iostream>
#include <chrono>

#include "cudabase.h"


#define DIM 800
#define ITERATION 1024

struct CUDAMemoryWorker {
    cudaStream_t stream;
    bool processing;
    int batch;
    unsigned char* M;
    unsigned char* dev_M;
};

double diffmsec(struct timeval  a,  struct timeval  b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);

    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ (double)usec/1000.0);
}

__global__ void mandel(int batch, int batch_size, int dim, double init_a, double init_b, double step, int niter, unsigned char *M) {
	
    int threadIdGlobal = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i_in_batch = floor((double)threadIdGlobal/dim);
    int i = batch * batch_size + i_in_batch; //global i
    int j = threadIdGlobal - i_in_batch*dim;
    if (i < dim && j < dim) {
        double im = init_b + (step * i);
        double cr;
        double a = cr = init_a + step * j;
        double b = im;
        int k = 0;
        for (k = 0; k < niter; k++) {
            double a2 = a*a;
            double b2 = b*b;
            if ((a2+b2) > 4.0) break;
            b = 2*a*b+im;
            a = a2-b2+cr;
        }
        M[i_in_batch*dim+j] = (unsigned char)255-((k*255 / niter));
    }
}


int main(int argc, char **argv) {
    double init_a=-2.125,init_b=-1.5,range=3.0;
    int dim = DIM, niter = ITERATION;
    // stats
    struct timeval t1,t2;
    int retries=1;
    double avg = 0;
    int batch_size = 0;
    int num_gpus = 0;
    int workers = 1;

    if (argc<6) {
        printf("Usage: %s size niterations retries workers batch_size [num_gpus]\n", argv[0]);
        printf("    num_gpus : Number of GPUs that should be used. If not informed, use all available GPUs\n\n");
        return 1;
    }
    else {
        dim = atoi(argv[1]);
        niter = atoi(argv[2]);
        retries = atoi(argv[3]);
        workers = atoi(argv[4]);
        batch_size = atoi(argv[5]);
        if (argc > 6) {
            num_gpus = atoi(argv[6]);
        }
    }

#if !defined(NO_DISPLAY)
    if (!num_gpus) {
        printf("Using all available GPUs\n");
    } else {
        printf("Using %d GPUs\n", num_gpus);
    }
#endif
    
    double * runs = (double *) malloc(retries*sizeof(double));

    double step = range/((double) dim);

#if !defined(NO_DISPLAY)
    SetupXWindows(dim,dim,1,NULL,"Mandelbroot Streaming");
#endif

    int total_devices;

    printf("bin;size;numiter;time (ms);workers;batch size\n");
    for (int r=0; r<retries; r++) {

        if (!num_gpus) {
            CudaSafeCall( cudaGetDeviceCount(&total_devices) );
            if (total_devices < 1) {
                printf("No CUDA-enabled device found");
                return 1;
            }
        } else {
            total_devices = num_gpus;
        }
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        int block_size = devProp.maxThreadsPerBlock; //max threads per block
        
        // Start time
        gettimeofday(&t1,NULL);

        int batches = ceil(dim/batch_size);

        int threads = block_size;
        int blocks = ceil((double)dim*batch_size / block_size);

        int total_workers = total_devices * workers;

        CUDAMemoryWorker* memWork = new CUDAMemoryWorker[total_workers];
        
        for (int w = 0; w < total_workers; w++) {
            int dev = w % total_devices;
            CudaSafeCall( cudaSetDevice(dev) );

			CudaSafeCall( cudaStreamCreate(&(memWork[w].stream)) );
            memWork[w].processing = false;
            memWork[w].batch = 0;
            CudaSafeCall( cudaHostAlloc((void**)&(memWork[w].M), dim * batch_size, cudaHostAllocDefault) );
            CudaSafeCall( cudaMalloc((void**)&(memWork[w].dev_M), dim * batch_size) );
        }

        CudaSafeCall( cudaSetDevice(0) );

#if !defined(NO_DISPLAY)
        printf("Starting %d batches with %d threads, on %d blocks\n", batches, threads, blocks);
#endif
        for(int b=0; b<batches; b++) {
            
            int w = b % total_workers; // if we have 2 GPUs, odd workers answers for GPU 1 and even for GPU 2
            int dev = w % total_devices;

            CudaSafeCall( cudaSetDevice(dev) );

            if (memWork[w].processing) {
                CudaSafeCall( cudaStreamSynchronize(memWork[w].stream) );
#if !defined(NO_DISPLAY)
                for (int i=0;i<batch_size;i++) {
                    ShowLine(&memWork[w].M[i*dim],dim,memWork[w].batch*batch_size+i);
                }
#endif
                memWork[w].processing = false;
                memWork[w].batch = 0;
            }
            
            mandel<<< blocks, threads, 0, memWork[w].stream >>>(b, batch_size, dim, init_a, init_b, step, niter, memWork[w].dev_M);

            CudaSafeCall( cudaMemcpyAsync(memWork[w].M, memWork[w].dev_M, dim * batch_size, cudaMemcpyDeviceToHost, memWork[w].stream) );

            memWork[w].processing = true;
            memWork[w].batch = b;

        }

        for (int w = 0; w < total_workers; w++) {
            int dev = w % total_devices;
            CudaSafeCall( cudaSetDevice(dev) );

            if (memWork[w].processing) {
                CudaSafeCall( cudaStreamSynchronize(memWork[w].stream) );
#if !defined(NO_DISPLAY)
                for (int i=0;i<batch_size;i++) {
                    ShowLine(&memWork[w].M[i*dim],dim,memWork[w].batch*batch_size+i);
                }
#endif
                memWork[w].processing = false;
                memWork[w].batch = 0;
            }

            CudaSafeCall( cudaFreeHost(memWork[w].M) );
            CudaSafeCall( cudaFree(memWork[w].dev_M) );

		    CudaSafeCall( cudaStreamDestroy(memWork[w].stream) );
        }

        delete memWork;

        // Stop time
        gettimeofday(&t2,NULL);

        avg += runs[r] = diffmsec(t2,t1);
        printf("%s (%d GPU);%d;%d;%.2f;%d;%d\n", argv[0], total_devices, dim, niter, runs[r], workers, batch_size);
    }

    avg = avg / (double) retries;
    double var = 0;
    for (int r=0; r<retries; r++) {
        var += (runs[r] - avg) * (runs[r] - avg);
    }
    var /= retries;

#if !defined(NO_DISPLAY)
    getchar();
    CloseXWindows();
#endif

    return 0;
}
