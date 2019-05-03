
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

unsigned int block_size = 32;
int total_devices = 1;
OCLDeviceObjects* gpus = NULL;

int workers = 1; 
int batch_size = 100; 
#define DIM 800
 
#define ITERATION 1024


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

double diff(struct timeval a,struct timeval b) {
	long sec = (a.tv_sec-b.tv_sec); 
	long usec = (a.tv_usec-b.tv_usec); 
	if(usec < 0)
	{
		--sec; 
		usec += 1000000;
	} 
	return ((double)(sec*1000)+(double)usec / 1000.0);
} 
#include <ff/pipeline.hpp>
 
#include <ff/farm.hpp>
 
using namespace ff; 
namespace spar{
	static inline ssize_t get_mac_core() {
		ssize_t n = 1; 
		FILE * f; 
		f = popen("cat /proc/cpuinfo |grep processor | wc -l","r"); 
		if(fscanf(f,"%ld",& n) == EOF)
		{
			pclose (f); 
			return n;
		} 
		pclose (f); 
		return n;
	} 
	static inline ssize_t get_env_num_workers() {
		ssize_t n = 1; 
		FILE * f; 
		f = popen("echo $SPAR_NUM_WORKERS","r"); 
		if(fscanf(f,"%ld",& n) == EOF)
		{
			pclose (f); 
			return n;
		} 
		pclose (f); 
		return n;
	} 
	static inline ssize_t get_Num_Workers() {
		ssize_t w_size = get_env_num_workers(); 
		if(w_size > 0)
		{
			return w_size;
		} 
		return get_mac_core();
	}
} 
struct _struct_spar0{
	_struct_spar0(int batch,int batch_size,int dim,double init_a,double init_b,double step,int niter,unsigned char * M,cl_mem dev_M, cl_kernel kernel, cl_command_queue queue, cl_event event) 
		: batch(batch),batch_size(batch_size),dim(dim),init_a(init_a),init_b(init_b),step(step),niter(niter),M(M),dev_M(dev_M),kernel(kernel),queue(queue),event(event) {
	}
	; 
	int batch; 
	int batch_size; 
	int dim; 
	double init_a; 
	double init_b; 
	double step; 
	int niter; 
	unsigned char * M;
	cl_mem dev_M;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_event event;
}; 
_struct_spar0 * _Stage_spar00(_struct_spar0 * _Input_spar,ff_node *const) {
	
	int dev = _Input_spar->batch % total_devices;

	int dimensions = 1;
	size_t Nblocks = ceil((double)_Input_spar->dim*_Input_spar->batch_size / block_size);
	// Number of work-items in each local work group, must be divisible by globalSize
	size_t localSize[1] = { block_size };
	// Total number of work-items
	size_t globalSize[1] = { block_size*Nblocks };

	cl_event evt;
	OpenCLCheckError( clEnqueueNDRangeKernel(_Input_spar->queue, _Input_spar->kernel, dimensions, NULL, globalSize, localSize, 0, NULL, &evt) );
	
	OpenCLCheckError( clEnqueueReadBuffer(_Input_spar->queue, _Input_spar->dev_M, CL_FALSE, 0, _Input_spar->dim * _Input_spar->batch_size, _Input_spar->M, 1, &evt, &(_Input_spar->event)) );
	
	// for(int i_batch = 0; i_batch < _Input_spar -> batch_size;i_batch++)
	// {
		
	// 	for(int j = 0; j < _Input_spar -> dim;j++)
	// 	{
	// 		int i = _Input_spar -> batch*_Input_spar -> batch_size+i_batch; 
	// 		double im = _Input_spar -> init_b+(_Input_spar -> step*i); 
	// 		double cr; 
	// 		double a = cr = _Input_spar -> init_a+_Input_spar -> step*j; 
	// 		double b = im; 
	// 		int k = 0; 
			
	// 		for(k = 0; k < _Input_spar -> niter;k++)
	// 		{
	// 			double a2 = a*a; 
	// 			double b2 = b*b; 
	// 			if((a2+b2) > 4.0)
	// 			break; 
	// 			b = 2*a*b+im; 
	// 			a = a2-b2+cr;
	// 		} 
	// 		_Input_spar -> M[i_batch*_Input_spar -> dim+j] = (unsigned char)255-((k*255 / _Input_spar -> niter));
	// 	}
	// } 

	return _Input_spar;
} 
_struct_spar0 * _Stage_spar01(_struct_spar0 * _Input_spar,ff_node *const) {
	{
		int dev = _Input_spar -> batch % total_devices;
		
		if (_Input_spar -> event != NULL) {
			OpenCLCheckError( clWaitForEvents(1, &(_Input_spar -> event)) );
		}
		#if !defined(NO_DISPLAY)
		
		for(int i = 0; i < _Input_spar -> batch_size;i++)
		{
			ShowLine(& _Input_spar -> M[i*_Input_spar -> dim], _Input_spar -> dim, _Input_spar -> batch*_Input_spar -> batch_size+i);
		} 
		#endif

		_Input_spar -> event = NULL;
 
		delete _Input_spar->M;
		OpenCLCheckError( clReleaseMemObject(_Input_spar -> dev_M) );
		OpenCLCheckError( clReleaseKernel(_Input_spar -> kernel) );
		OpenCLCheckError( clReleaseCommandQueue(_Input_spar -> queue) );
	} 
	delete _Input_spar; 
	return (_struct_spar0 *)GO_ON;
} 
struct _ToStream_spar0 : ff_node_t < _struct_spar0 >{
	int batches; 
	int dim; 
	double init_a; 
	double init_b; 
	double step; 
	int niter; 
	int batch_size; 
	_struct_spar0 * svc(_struct_spar0 * _Input_spar) {
		
		cl_int status;
		
		for(int batch = 0; batch < batches;batch++)
		{
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
			
			_struct_spar0 * stream_spar = new _struct_spar0 (batch,batch_size,dim,init_a,init_b,step,niter,M,dev_M,kernel,queue,NULL); 
			ff_send_out (stream_spar); 
			;
		} 
		return EOS;
	}
}; 
void mandel(int block_size,int dim,double init_b,double init_a,int niter,double step) {
	_ToStream_spar0 _ToStream_spar0_call; 
	ff_Farm < _struct_spar0 > _Stage_spar00_call(_Stage_spar00,workers); 
	_Stage_spar00_call.add_emitter(_ToStream_spar0_call); 
	ff_node_F < _struct_spar0 > _Stage_spar01_call (_Stage_spar01); 
	_Stage_spar00_call.add_collector(_Stage_spar01_call); 
	int batches = ceil(dim / batch_size); 
	_ToStream_spar0_call.batches = batches; 
	_ToStream_spar0_call.dim = dim; 
	_ToStream_spar0_call.init_a = init_a; 
	_ToStream_spar0_call.init_b = init_b; 
	_ToStream_spar0_call.step = step; 
	_ToStream_spar0_call.niter = niter; 
	_ToStream_spar0_call.batch_size = batch_size;

	if(_Stage_spar00_call.run_and_wait_end() < 0)
	{
		error("Running farm\n"); 
		exit(1);
	}
} 
int main(int argc,char * * argv) {
	double init_a = - 2.125,init_b = - 1.5,range = 3.0; 
	int dim = DIM,niter = ITERATION; 
	struct timeval t1,t2; 
	int retries = 1; 
	double avg = 0; 
    int num_gpus = 0;

    cl_int status;

	if(argc < 6)
	{
		printf("Usage: %s size niterations retries workers batch_size [num_gpus]\n", argv[0]); 
        printf("    num_gpus : Number of GPUs that should be used. If not informed, use all available GPUs\n\n");
		return 0;
	} else 
	{
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

	double * runs = (double *)malloc(retries*sizeof(double)); 
	double step = range / ((double)dim); 
	#if !defined(NO_DISPLAY)
 
	SetupXWindows(dim,dim,1,NULL,"Sequential Mandelbroot"); 
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

	for(int r = 0; r < retries;r++)
	{
		gettimeofday(& t1,NULL); 
		mandel(block_size,dim,init_b,init_a,niter,step); 
		gettimeofday(& t2,NULL); 
		avg += runs[r] = diff(t2,t1); 
		printf("%s (%d GPU);%d;%d;%.2f;%d;%d\n",argv[0],total_devices,dim,niter,runs[r],workers,batch_size);
	} 
	avg = avg / (double)retries; 
	double var = 0; 
	
	for(int r = 0; r < retries;r++)
	{
		var += (runs[r]-avg)*(runs[r]-avg);
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
