
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

int workers = 1; 
int batch_size = 100; 
#define DIM 800
 
#define ITERATION 1024

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
	_struct_spar0(int batch,int batch_size,int dim,double init_a,double init_b,double step,int niter,unsigned char * M,unsigned char * dev_M, cudaStream_t cuda_stream) 
		: batch(batch),batch_size(batch_size),dim(dim),init_a(init_a),init_b(init_b),step(step),niter(niter),M(M),dev_M(dev_M),cuda_stream(cuda_stream) {
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
	unsigned char * dev_M;
	cudaStream_t cuda_stream;
}; 
_struct_spar0 * _Stage_spar00(_struct_spar0 * _Input_spar,ff_node *const) {
	
	CudaSafeCall( cudaSetDevice(_Input_spar->batch % total_devices) );

	int threads = block_size;
	int blocks = ceil((double)_Input_spar->dim*_Input_spar->batch_size / block_size);

	mandel_kernel<<< blocks, threads, 0, _Input_spar -> cuda_stream >>>(_Input_spar->batch, _Input_spar->batch_size, _Input_spar->dim, _Input_spar->init_a, _Input_spar->init_b, _Input_spar->step, _Input_spar->niter, _Input_spar->dev_M);

	CudaSafeCall( cudaMemcpyAsync(_Input_spar->M, _Input_spar->dev_M, _Input_spar->dim * _Input_spar->batch_size, cudaMemcpyDeviceToHost, _Input_spar->cuda_stream) ); //unsigned char is 1 byte, so its not necessary to call sizeof

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
		CudaSafeCall( cudaSetDevice(_Input_spar -> batch % total_devices) );
		
		CudaSafeCall( cudaStreamSynchronize(_Input_spar -> cuda_stream) );

		#if !defined(NO_DISPLAY)
		
		for(int i = 0; i < _Input_spar -> batch_size;i++)
		{
			ShowLine(& _Input_spar -> M[i*_Input_spar -> dim], _Input_spar -> dim, _Input_spar -> batch*_Input_spar -> batch_size+i);
		} 
		#endif
 
    	CudaSafeCall( cudaFreeHost(_Input_spar -> M) );
		CudaSafeCall( cudaFree(_Input_spar -> dev_M) );

		CudaSafeCall( cudaStreamDestroy(_Input_spar -> cuda_stream) );

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
		
		for(int batch = 0; batch < batches;batch++)
		{
			CudaSafeCall( cudaSetDevice(batch % total_devices) );

			cudaStream_t cuda_stream;
			CudaSafeCall( cudaStreamCreate(&cuda_stream) );

			unsigned char * M; 
    		CudaSafeCall( cudaHostAlloc((void**)&M, dim * batch_size, cudaHostAllocDefault) );
			
			unsigned char *dev_M; //on device
			CudaSafeCall( cudaMalloc((void**)&dev_M, dim * batch_size) ); //unsigned char is 1 byte, so its not necessary to call sizeof
			
			_struct_spar0 * stream_spar = new _struct_spar0 (batch,batch_size,dim,init_a,init_b,step,niter,M,dev_M,cuda_stream); 
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
	int batches = ceil((double)dim / batch_size); 
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

	if(argc < 6)
	{
		printf("Usage: %s size niterations retries workers batch_size [num_gpus]\n", argv[0]); 
        printf("    num_gpus : Number of GPUs that should be used. If not informed, use all available GPUs\n\n");
        exit(-1);
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
	#if !defined(NO_DISPLAY)
 
	printf("Average on %d experiments = %f (ms) Std. Dev. %f\n\nPress a key\n",retries,avg,sqrt(var)); 
	getchar(); 
	CloseXWindows(); 
	#endif
 
	return 0;
}
