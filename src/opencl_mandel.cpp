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

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#if !defined(NO_DISPLAY)
#include "marX2.h"
#endif
#include <sys/time.h>
#include "math.h"
#include <cstring>

#include <iostream>
#include <chrono>

#include <CL/opencl.h>

#include "oclbase.h"

const char* kernelSource = 
"__kernel void mandel_kernel(int batch, int batch_size, int dim, double init_a, double init_b, double step, int niter, __global unsigned char *M) { \n"
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

#define DIM 800
#define ITERATION 1024

struct OCLDeviceObjects {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
};
struct OCLMemoryWorker {
    OCLDeviceObjects* gpu;
    bool processing;
    int batch;
    unsigned char* M;
    cl_mem dev_M;
    cl_command_queue queue;
    cl_event event;
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
        workers = atoi(argv[4]);
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
    SetupXWindows(dim,dim,1,NULL,"Mandelbroot Streaming");
#endif

    printf("bin;size;numiter;time (ms);workers;batch size\n");
    for (int r=0; r<retries; r++) {

        cl_device_id* devices = NULL;
        int total_devices = 0;
        getGPUs(&devices, &total_devices);
        if (num_gpus) {
            total_devices = num_gpus;
        }

#if !defined(NO_DISPLAY)
        printf("Found %d OpenCL GPUs\n", total_devices);
        for (int dev=0; dev<total_devices; dev++) {
            size_t valueSize;
            OpenCLCheckError( clGetDeviceInfo(devices[dev], CL_DEVICE_NAME, 0, NULL, &valueSize) );
            char* name = (char*) malloc(valueSize);
            OpenCLCheckError( clGetDeviceInfo(devices[dev], CL_DEVICE_NAME, valueSize, name, NULL) );
            printf("  OpenCL Device #%d: \"%s\"\n", dev+1, name);
        }
#endif

        int total_workers = total_devices * workers;

        OCLDeviceObjects* gpus = new OCLDeviceObjects[total_devices];

        for (int dev = 0; dev < total_devices; dev++) {

            gpus[dev].device = devices[dev];

            gpus[dev].context = clCreateContext(NULL, 1, &gpus[dev].device, NULL, NULL, &status);
            OpenCLCheckError(status);

            gpus[dev].program = clCreateProgramWithSource(gpus[dev].context, 1, (const char**)&kernelSource, NULL, &status);
            OpenCLCheckError(status);
            
            status = clBuildProgram(gpus[dev].program, 1, &gpus[dev].device, NULL, NULL, NULL);
            OpenCLCheckBuildError(status, gpus[dev].program, gpus[dev].device);

            gpus[dev].kernel = clCreateKernel(gpus[dev].program, "mandel_kernel", &status);
            OpenCLCheckError(status);

            int i_parm = 1;
            OpenCLCheckError( clSetKernelArg(gpus[dev].kernel, i_parm++, sizeof(int), &batch_size) );
            OpenCLCheckError( clSetKernelArg(gpus[dev].kernel, i_parm++, sizeof(int), &dim) );
            OpenCLCheckError( clSetKernelArg(gpus[dev].kernel, i_parm++, sizeof(double), &init_a) );
            OpenCLCheckError( clSetKernelArg(gpus[dev].kernel, i_parm++, sizeof(double), &init_b) );
            OpenCLCheckError( clSetKernelArg(gpus[dev].kernel, i_parm++, sizeof(double), &step) );
            OpenCLCheckError( clSetKernelArg(gpus[dev].kernel, i_parm++, sizeof(int), &niter) );
        }

        OCLMemoryWorker* memWork = new OCLMemoryWorker[total_workers];

        for (int w = 0; w < total_workers; w++) {
            memWork[w].gpu = &gpus[w % total_devices];
            memWork[w].processing = false;
            memWork[w].batch = 0;
            memWork[w].M = new unsigned char [dim * batch_size]; 
            memWork[w].dev_M = clCreateBuffer(memWork[w].gpu->context, CL_MEM_WRITE_ONLY, dim * batch_size, NULL, &status);
            OpenCLCheckError(status);
            memWork[w].queue = clCreateCommandQueue(memWork[w].gpu->context, memWork[w].gpu->device, 0, &status);
            OpenCLCheckError(status);
            memWork[w].event = NULL;
        }

        // Start time
        gettimeofday(&t1,NULL);

        size_t workGroupSize;
        clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, NULL);
        size_t block_size = workGroupSize; //max threads per block

        int batches = ceil(dim/batch_size);

        int dimensions = 1;
        size_t Nblocks = ceil((double)(dim*batch_size) / block_size);
        // Number of work-items in each local work group, must be divisible by globalSize
        size_t localSize[1] = { block_size };
        // Total number of work-items
        size_t globalSize[1] = { block_size*Nblocks };

#if !defined(NO_DISPLAY)
        printf("Will run %d batches of %lu threads, on a total of %lu threads in blocks\n", batches, localSize[0], globalSize[0]);
#endif

        for(int b=0; b<batches; b++) {
            
            int w = b % total_workers; // if we have 2 GPUs, odd workers answers for GPU 1 and even for GPU 2

            if (memWork[w].processing) {
                OpenCLCheckError( clWaitForEvents(1, &memWork[w].event) );
#if !defined(NO_DISPLAY)
                for (int i=0;i<batch_size;i++) {
                    ShowLine(&memWork[w].M[i*dim],dim,memWork[w].batch*batch_size+i);
                }
#endif
                memWork[w].processing = false;
                memWork[w].batch = 0;
                memWork[w].event = NULL;
            }

            OpenCLCheckError( clSetKernelArg(memWork[w].gpu->kernel, 0, sizeof(int), &b) );
            OpenCLCheckError( clSetKernelArg(memWork[w].gpu->kernel, 7, sizeof(cl_mem), &memWork[w].dev_M) );

            cl_event evt;
            OpenCLCheckError( clEnqueueNDRangeKernel(memWork[w].queue, memWork[w].gpu->kernel, dimensions, NULL, globalSize, localSize, 0, NULL, &evt) );

            OpenCLCheckError( clEnqueueReadBuffer(memWork[w].queue, memWork[w].dev_M, CL_FALSE, 0, dim * batch_size, memWork[w].M, 1, &evt, &memWork[w].event) );
            
            memWork[w].processing = true;
            memWork[w].batch = b;

        }

        for (int w = 0; w < total_workers; w++) {
            if (memWork[w].processing) {
                OpenCLCheckError( clWaitForEvents(1, &memWork[w].event) );

#if !defined(NO_DISPLAY)
                for(int i = 0; i < batch_size; i++) {
                    ShowLine(&memWork[w].M[i*dim], dim, memWork[w].batch*batch_size+i);
                }
#endif
                memWork[w].processing = false;
                memWork[w].batch = 0;
                memWork[w].event = NULL;
            }

            delete memWork[w].M;
            OpenCLCheckError( clReleaseMemObject(memWork[w].dev_M) );
            OpenCLCheckError( clReleaseCommandQueue(memWork[w].queue) );
        }
        delete memWork;

        for (int dev = 0; dev < total_devices; dev++) {
            OpenCLCheckError( clReleaseKernel(gpus[dev].kernel) );
            OpenCLCheckError( clReleaseProgram(gpus[dev].program) );
            OpenCLCheckError( clReleaseContext(gpus[dev].context) );
        }
        delete gpus;

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
    printf("Average on %d experiments = %f (ms) Std. Dev. %f\n\nPress a key\n",retries,avg,sqrt(var));
    getchar();
    CloseXWindows();
#endif

    return 0;
}
