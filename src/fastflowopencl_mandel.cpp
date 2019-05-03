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
#include <cstring>

#include <CL/opencl.h>
#include "oclbase.h"
struct OCLDeviceObjects {
    cl_device_id device;
    cl_context context;
    cl_program program;
};
OCLDeviceObjects* gpus = NULL;

const char* kernelSource = 
"__kernel void mandel_kernel(int batch, int batch_size, int dim, double init_a, double init_b, \n"
"       double step, int niter, __global unsigned char *M) { \n"
"   int threadIdGlobal = get_global_id(0); \n"
"   int i_in_batch = floor((double)threadIdGlobal/dim); \n"
"   int i = batch * batch_size + i_in_batch; \n" //global i
"   int j = threadIdGlobal - i_in_batch*dim; \n"
"   if (i < dim && j < dim) { \n"
"       double im=init_b+(step*i); \n"
"       double cr; \n"
"       double a = cr = init_a + step * j; \n"
"       double b = im; \n"
"       int k = 0; \n"
"       for (k=0; k<niter; k++) { \n"
"           double a2=a*a; \n"
"           double b2=b*b; \n"
"           if ((a2+b2)>4.0) break; \n"
"           b=2*a*b+im; \n"
"           a=a2-b2+cr; \n"
"       } \n"
"       M[i_in_batch*dim+j]= (unsigned char) 255-((k*255/niter)); \n"
"   } \n"
"}";

void getGPUs(cl_device_id** devices, int* total_devices) {
    int total = 0;
    
    cl_uint platformCount;
    OpenCLCheckError( clGetPlatformIDs(0, NULL, &platformCount) );

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    OpenCLCheckError( clGetPlatformIDs(platformCount, platforms, NULL) );

    for (int i = 0; i < platformCount; ++i) {
        cl_uint deviceCount;
        OpenCLCheckError( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount) );
        
        cl_device_id* dev_aux = new cl_device_id[total + deviceCount];
        if (total > 0) {
            memcpy(dev_aux, **devices, total);
        }
        if (*devices) delete *devices;
        *devices = dev_aux;

        OpenCLCheckError( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, &((*devices)[total]), NULL) );
        total += deviceCount;
    }
    
    *total_devices = total;
}
 
unsigned int block_size = 32;
int total_devices = 1;

int batch_size = 32; 

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
    task_t(int batch, unsigned char *M, cl_mem dev_M, cl_kernel kernel, cl_command_queue queue, cl_event event) :
        batch(batch), M(M), dev_M(dev_M), kernel(kernel), queue(queue), event(event){};
    int batch;
    unsigned char* M;
	cl_mem dev_M;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_event event;
};
class Emitter: public ff_node_t<task_t> {
public:
    int dim;
	double init_a; 
	double init_b; 
	double step; 
	int niter; 
    Emitter(int dim, int niter, double init_a, double init_b, double step ): dim(dim), niter(niter), init_a(init_a), init_b(init_b), step(step) {}
    task_t *svc(task_t*) {
		cl_int status;
    	
        int batches = ceil((double)dim / batch_size); 
		for(int batch = 0; batch < batches;batch++) {

			int dev = batch % total_devices;

			unsigned char * M = new unsigned char [dim*batch_size]; 

			cl_mem dev_M = clCreateBuffer(gpus[dev].context, CL_MEM_WRITE_ONLY, dim * batch_size, NULL, &status);
			OpenCLCheckError(status);

			// cl_kernel objects are not thread-safe (pg 360 from OpenCL 1.2 Specification)
			cl_kernel kernel = clCreateKernel(gpus[dev].program, "mandel_kernel", &status);
			OpenCLCheckError(status);

			int i_parm = 0;
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &batch) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &batch_size) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &dim) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(double), &init_a) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(double), &init_b) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(double), &step) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &niter) );
			OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &dev_M) );

            cl_command_queue queue = clCreateCommandQueue(gpus[dev].context, gpus[dev].device, 0, &status);
            OpenCLCheckError(status);

            ff_send_out(new task_t(batch, M, dev_M, kernel, queue, NULL));
        }
        return (task_t*)EOS;
    }
};

class Worker: public ff_node_t<task_t> {
public:
    int dim;
    Worker(int dim): dim(dim) {}
    task_t* svc(task_t* t) {

	    int dev = t->batch % total_devices;

        int dimensions = 1;
        size_t Nblocks = ceil((double)dim*batch_size / block_size);
        // Number of work-items in each local work group, must be divisible by globalSize
        size_t localSize[1] = { block_size };
        // Total number of work-items
        size_t globalSize[1] = { block_size*Nblocks };

    	cl_event evt;
	    OpenCLCheckError( clEnqueueNDRangeKernel(t->queue, t->kernel, dimensions, NULL, globalSize, localSize, 0, NULL, &evt) );

    	OpenCLCheckError( clEnqueueReadBuffer(t->queue, t->dev_M, CL_FALSE, 0, dim * batch_size, t->M, 1, &evt, &(t->event)) );
        
        return t;
    }
};

class Collector: public ff_node_t<task_t> {
public:
    int dim;
    Collector(int dim): dim(dim) {};
	task_t* svc(task_t* t) {

	    int dev = t->batch % total_devices;
		if (t->event != NULL) {
			OpenCLCheckError( clWaitForEvents(1, &(t->event)) );
		}

#if !defined(NO_DISPLAY)
		for(int i = 0; i < batch_size; i++) {
			ShowLine(&t->M[i*dim], dim, t->batch*batch_size+i);
		} 
#endif
		t->event = NULL;

        delete t->M;
		OpenCLCheckError( clReleaseMemObject(t->dev_M) );
		OpenCLCheckError( clReleaseKernel(t->kernel) );
		OpenCLCheckError( clReleaseCommandQueue(t->queue) );

        delete t;
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
    cl_int status;

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
    if (!num_gpus) {
        printf("Using all available GPUs\n");
    } else {
        printf("Using %d GPUs\n", num_gpus);
    }

    SetupXWindows(dim,dim,1,NULL,"Streaming Mandelbroot");
#endif

    printf("bin;size;numiter;time (ms);workers;batch size\n");
	
	cl_device_id* devices = NULL;
	getGPUs(&devices, &total_devices);
	if (num_gpus) {
		total_devices = num_gpus;
	}
	if (total_devices < 1) {
		printf("No OpenCL-enabled device found");
		return 1;
	}

	gpus = new OCLDeviceObjects[total_devices];

	for (int dev = 0; dev < total_devices; dev++) {
		gpus[dev].device = devices[dev];

		gpus[dev].context = clCreateContext(NULL, 1, &gpus[dev].device, NULL, NULL, &status);
		OpenCLCheckError(status);

		gpus[dev].program = clCreateProgramWithSource(gpus[dev].context, 1, (const char**)&kernelSource, NULL, &status);
		OpenCLCheckError(status);
		
		status = clBuildProgram(gpus[dev].program, 1, &gpus[dev].device, NULL, NULL, NULL);
		OpenCLCheckBuildError(status, gpus[dev].program, gpus[dev].device);
	}
	
	size_t workGroupSize;
	clGetDeviceInfo(gpus[0].device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, NULL);
	block_size = workGroupSize; //max threads per block

    for (int r=0; r<retries; r++) {

        // Start time
        gettimeofday(&t1,NULL);

        std::vector<ff_node*> workers;
        for (int w=0; w<n_workers; w++) {
            workers.push_back(new Worker(dim));
        }

        ff_farm<> farm;
        farm.add_emitter(new Emitter(dim, niter, init_a, init_b, step));
        farm.add_workers(workers);
        farm.add_collector(new Collector(dim));
        int ret = farm.run_and_wait_end();
        if (ret < 0) {
            printf("Error on FastFlow farm: %d\n", ret);
            return -1;
        }

        // Stop time
        gettimeofday(&t2,NULL);

        avg += runs[r] = diff(t2,t1);
		printf("%s (%d GPU);%d;%d;%.2f;%d;%d\n", argv[0], total_devices, dim, niter, runs[r], n_workers, batch_size);
    }
    avg = avg / (double) retries;
    double var = 0;
    for (int r=0; r<retries; r++) {
        var += (runs[r] - avg) * (runs[r] - avg);
    }
    var /= retries;

	for (int dev = 0; dev < total_devices; dev++) {
		OpenCLCheckError( clReleaseProgram(gpus[dev].program) );
		OpenCLCheckError( clReleaseContext(gpus[dev].context) );
	}
	delete gpus;

#if !defined(NO_DISPLAY)
    printf("Average on %d experiments = %f (ms) Std. Dev. %f\n\nPress a key\n",retries,avg,sqrt(var));
    getchar();
    CloseXWindows();
#endif

    return 0;
}
