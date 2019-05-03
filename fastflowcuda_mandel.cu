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
int block_size = 32;
int total_devices = 1;

int batch_size = 1; 

__global__ void mandel_kernel(int batch, int batch_size, int dim, double init_a, double init_b, double step, int niter, unsigned char *M) {

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

#include <ff/farm.hpp>
using namespace ff; 

#define DIM 800
#define ITERATION 1024

double diff(struct timeval  a,  struct timeval  b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);

    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ (double)usec/1000.0);
}

struct task_t {
    task_t(int batch, unsigned char *M, unsigned char* dev_M, cudaStream_t cuda_stream) : batch(batch), M(M), dev_M(dev_M), cuda_stream(cuda_stream){};
    int batch;
    unsigned char* M;
	unsigned char* dev_M;
	cudaStream_t cuda_stream;
};
class Emitter: public ff_node_t<task_t> {
public:
    int dim;
    Emitter(int dim): dim(dim) {}
    task_t *svc(task_t*) {
    	int batches = ceil((double)dim / batch_size); 
		for(int batch = 0; batch < batches;batch++) {
			CudaSafeCall( cudaSetDevice(batch % total_devices) );

			unsigned char * M; 
    		CudaSafeCall( cudaHostAlloc((void**)&M, dim * batch_size, cudaHostAllocDefault) );

			unsigned char *dev_M; //on device
			CudaSafeCall( cudaMalloc((void**)&dev_M, dim * batch_size) );

			cudaStream_t cuda_stream;
			CudaSafeCall( cudaStreamCreate(&cuda_stream) );

            ff_send_out(new task_t(batch, M, dev_M, cuda_stream));
        }
        return (task_t*)EOS;
    }
};

class Worker: public ff_node_t<task_t> {
public:
    int dim;
    int niter;
    double init_a;
    double init_b;
    double step;
    Worker(int dim, int niter, double init_a, double init_b, double step): dim(dim), niter(niter), init_a(init_a), init_b(init_b), step(step) {}
    task_t* svc(task_t* t) {

	    CudaSafeCall( cudaSetDevice(t->batch % total_devices) );

        int threads = block_size;
        int blocks = ceil((double)dim*batch_size / block_size);
        
    	mandel_kernel<<< blocks, threads, 0, t->cuda_stream >>>(t->batch, batch_size, dim, init_a, init_b, step, niter, t->dev_M);

    	CudaSafeCall( cudaMemcpyAsync(t->M, t->dev_M, dim * batch_size, cudaMemcpyDeviceToHost, t->cuda_stream) );

        return t;
    }
};

class Collector: public ff_node_t<task_t> {
public:
    int dim;
    Collector(int dim): dim(dim) {};
	task_t* svc(task_t* t) {

	    CudaSafeCall( cudaSetDevice(t->batch % total_devices) );
		CudaSafeCall( cudaStreamSynchronize(t->cuda_stream) );

#if !defined(NO_DISPLAY)
		for(int i = 0; i < batch_size; i++) {
			ShowLine(&t->M[i*dim], dim, t->batch*batch_size+i);
		} 
#endif
        CudaSafeCall( cudaFreeHost(t->M) );
        CudaSafeCall( cudaFree(t->dev_M) );
		CudaSafeCall( cudaStreamDestroy(t->cuda_stream) );

    	return (task_t*)GO_ON;
	}
};

int main(int argc, char **argv) {
    double init_a=-2.125,init_b=-1.5,range=3.0;
    int dim = DIM, niter = ITERATION;
    // stats
    struct timeval t1,t2;
    int retries=1;
    double avg = 0;
    int n_workers = 1;
    int num_gpus = 0;

    if (argc<6) {
		printf("Usage: %s size niterations retries workers batch_size [num_gpus]\n", argv[0]); 
        printf("    num_gpus : Number of GPUs that should be used. If not informed, use all available GPUs\n\n");
        exit(-1);
    }
    else {
        dim = atoi(argv[1]);
        niter = atoi(argv[2]);
        retries = atoi(argv[3]);
        n_workers = atoi(argv[4]);
		batch_size = atoi(argv[5]);
        if (argc > 6) {
            num_gpus = atoi(argv[6]);
        }
    }
    double * runs = (double *) malloc(retries*sizeof(double));

    double step = range/((double) dim);

#if !defined(NO_DISPLAY)
    SetupXWindows(dim,dim,1,NULL,"Sequential Mandelbroot");
#endif

    printf("bin;size;numiter;time (ms);workers;batch size\n");

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
	block_size = devProp.maxThreadsPerBlock; //max threads per block

    for (int r=0; r<retries; r++) {

        // Start time
        gettimeofday(&t1,NULL);

        std::vector<ff_node*> workers;
        for (int w=0; w<n_workers; w++) {
            workers.push_back(new Worker(dim, niter, init_a, init_b, step));
        }

        ff_farm<> farm;
        farm.add_emitter(new Emitter(dim));
        farm.add_workers(workers);
        farm.add_collector(new Collector(dim));
        int ret = farm.run_and_wait_end();
        if (ret < 0) {
            printf("Error on FastFlow farm: %d\n", ret);
            return -1;
        }

        // Stop time
        gettimeofday(&t2,NULL);

        CudaCheckError();
        
        avg += runs[r] = diff(t2,t1);
		printf("%s (%d GPU);%d;%d;%.2f;%d;%d\n", argv[0], total_devices, dim, niter, runs[r], n_workers, batch_size);
    }
    avg = avg / (double) retries;
    double var = 0;
    for (int r=0; r<retries; r++) {
        var += (runs[r] - avg) * (runs[r] - avg);
    }
    var /= retries;

#if !defined(NO_DISPLAY)
    printf("Average on %d experiments = %f (ms) Std. Dev. %f\n\nPress a key\n",retries,avg,sqrt(var));
    getchar();
    CloseXWindows();
#endif

    return 0;
}
