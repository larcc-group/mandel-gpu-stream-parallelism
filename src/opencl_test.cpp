#define CL_TARGET_OPENCL_VERSION 120

//https://www.pgroup.com/userforum/viewtopic.php?p=22545&sid=832bf90dfa27396f7b6239bd8a714530#22545
// #define __CUDACC__ 

#include <iostream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/time.h>

#include <CL/opencl.h>

// __global__ void test(int N) {
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;
//     // printf("%d: block %d, thread %d/%d\n", gid, blockIdx.x, threadIdx.x, blockDim.x-1);
//     printf("%d: Dim (%d, %d), Block (%d, %d), thread (%d, %d)\n", gid, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
// }

const char* kernelSource = 
"__kernel void test_kernel(__global int* N) { \n"
"   int threadIdGlobal = get_global_id(0) * get_global_size(0) + get_global_id(1); \n"
// "   int threadIdGlobal = get_global_id(0); \n"
"   printf(\"Thread %d (%lu, %lu)\\n\", threadIdGlobal, get_global_id(0), get_global_id(1)); \n"
"   if (threadIdGlobal == 0) { \n"
"       printf( \n"
"           \"\\nThread %d:\\n\"\n"
"           \"    dim %d: groups %lu, global %lu/%lu, local %lu/%lu\\n\"\n"
"           \"    dim %d: groups %lu, global %lu/%lu, local %lu/%lu\\n\"\n"
"           , threadIdGlobal \n"
"           , 0, get_num_groups(0), get_global_id(0)+1, get_global_size(0), get_local_id(0)+1, get_local_size(0) \n"
"           , 1, get_num_groups(1), get_global_id(1)+1, get_global_size(1), get_local_id(1)+1, get_local_size(1) \n"
"       ); \n"
"   } \n"
"}";

char* queryTextDeviceInfo(cl_device_id device, cl_device_info param_name) {
    //https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/clGetDeviceInfo.html
    size_t valueSize;
    clGetDeviceInfo(device, param_name, 0, NULL, &valueSize);
    char* value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, param_name, valueSize, value, NULL);
    return value;
}
char* queryTextPlatformInfo(cl_platform_id platform, cl_platform_info param_name) {
    //https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/clGetPlatformInfo.html
    size_t valueSize;
    clGetPlatformInfo(platform, param_name, 0, NULL, &valueSize);
    char * value = (char*) malloc(valueSize);
    clGetPlatformInfo(platform, param_name, valueSize, value, NULL);
    return value;
}

void checkOpenCLErrorCode(std::string func, cl_int status) {
    if (status != CL_SUCCESS) {
        printf("OpenCL ERROR on %s: %d\n", func.c_str(), status);
    }
}

int main(int argc, char const *argv[]){
    int N = 10;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
	printf("Starting OpenCL test, N = %d\n", N);

    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    printf("Found %d OpenCL platforms\n", platformCount);

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (int i = 0; i < platformCount; ++i) {

        char* name = queryTextPlatformInfo(platforms[i], CL_PLATFORM_NAME);
        printf("Platform #%d:        %s\n", i+1, name);
        free(name);
        
        char* version = queryTextPlatformInfo(platforms[i], CL_PLATFORM_VERSION);
        printf("  OpenCL version:   %s\n", version);
        free(version);

        cl_uint deviceCount;
        //CL_DEVICE_TYPE_GPU
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        printf("  Platform contains %d OpenCL devices\n", deviceCount);

        cl_device_id* devices = new cl_device_id[platformCount];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        for (int j = 0; j < deviceCount; j++) {
            printf("\n");

            cl_ulong clulong;
            cl_uint cluint;
            cl_device_type cldevtype;

            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cldevtype), &cldevtype, NULL);

            char* name = queryTextDeviceInfo(devices[j], CL_DEVICE_NAME);
            printf("  Platform's OpenCL Device #%d: \"%s\"\n", j+1, name);
            free(name);
            printf("    Device type:                        %s\n", (
                cldevtype == CL_DEVICE_TYPE_CPU ? "CPU" :
                cldevtype == CL_DEVICE_TYPE_GPU ? "GPU" :
                cldevtype == CL_DEVICE_TYPE_ACCELERATOR ? "Accelerator" :
                "Other"
            ));

            char* deviceVersion = queryTextDeviceInfo(devices[j], CL_DEVICE_VERSION);
            printf("    Hardware version:                   %s\n", deviceVersion);
            free(deviceVersion);

            char* driverVersion = queryTextDeviceInfo(devices[j], CL_DRIVER_VERSION);
            printf("    Software version:                   %s\n", driverVersion);
            free(driverVersion);

            char* openclVersion = queryTextDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION);
            printf("    OpenCL C version:                   %s\n", openclVersion);
            free(openclVersion);

            printf("    Memory:\n");
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(clulong), &clulong, NULL);
            printf("      Total global memory:              %lu MB\n", clulong/(1024*1024));
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(clulong), &clulong, NULL);
            printf("      Total local memory:               %lu KB\n", clulong/1024);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(clulong), &clulong, NULL);
            printf("      Total global memory cache:        %lu KB\n", clulong/1024);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cluint), &cluint, NULL);
            printf("    Compute units (multiprocessors):    %d\n", cluint);
            
            size_t workGroupSize;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, NULL);
            printf("    Max Work Group size:                %lu\n", workGroupSize);

            cl_uint dimensions;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dimensions), &dimensions, NULL);
            printf("                              Dimension (");
            for (int d = 0; d < dimensions; d++) printf("%s %d ", (d != 0 ? "x " : ""), d);
            printf(")\n");
            
            size_t* dimensionSizes = new size_t[dimensions];
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dimensionSizes), &dimensionSizes, NULL);
            printf("    Max Work Item sizes:                (");
            for (int d = 0; d < dimensions; d++) printf("%s%lu", (d != 0 ? " x " : ""), dimensionSizes[d]);
            printf(")\n");
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cluint), &cluint, NULL);
            printf("    Clock rate:                         %d MHz\n", cluint);
            
        }

        delete devices;
        
        printf("\n");
    }


    // std::cout << "Common error codes:" << std::endl;
    // std::cout << "CL_INVALID_COMMAND_QUEUE: " << CL_INVALID_COMMAND_QUEUE << std::endl;
    // std::cout << "CL_INVALID_CONTEXT: " << CL_INVALID_CONTEXT << std::endl;
    // std::cout << "CL_INVALID_MEM_OBJECT: " << CL_INVALID_MEM_OBJECT << std::endl;
    // std::cout << "CL_INVALID_VALUE: " << CL_INVALID_VALUE << std::endl;
    // std::cout << "CL_INVALID_EVENT_WAIT_LIST: " << CL_INVALID_EVENT_WAIT_LIST << std::endl;
    // std::cout << "CL_MISALIGNED_SUB_BUFFER_OFFSET: " << CL_MISALIGNED_SUB_BUFFER_OFFSET << std::endl;
    // std::cout << "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: " << CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST << std::endl;
    // std::cout << "CL_MEM_OBJECT_ALLOCATION_FAILURE: " << CL_MEM_OBJECT_ALLOCATION_FAILURE << std::endl;
    // std::cout << "CL_INVALID_OPERATION: " << CL_INVALID_OPERATION << std::endl;
    // std::cout << "CL_OUT_OF_RESOURCES: " << CL_OUT_OF_RESOURCES << std::endl;
    // std::cout << "CL_OUT_OF_HOST_MEMORY: " << CL_OUT_OF_HOST_MEMORY << std::endl;
	// std::cout << "CL_INVALID_KERNEL: " << CL_INVALID_KERNEL << std::endl;
    // std::cout << "CL_INVALID_ARG_INDEX: " << CL_INVALID_ARG_INDEX << std::endl;
    // std::cout << "CL_INVALID_ARG_VALUE: " << CL_INVALID_ARG_VALUE << std::endl;
    // std::cout << "CL_INVALID_MEM_OBJECT: " << CL_INVALID_MEM_OBJECT << std::endl;
    // std::cout << "CL_INVALID_SAMPLER: " << CL_INVALID_SAMPLER << std::endl;
    // std::cout << "CL_INVALID_ARG_SIZE: " << CL_INVALID_ARG_SIZE << std::endl;
    // std::cout << "CL_INVALID_ARG_VALUE: " << CL_INVALID_ARG_VALUE << std::endl;
    // std::cout << "CL_OUT_OF_RESOURCES: " << CL_OUT_OF_RESOURCES << std::endl;
    // std::cout << "CL_OUT_OF_HOST_MEMORY: " << CL_OUT_OF_HOST_MEMORY << std::endl;

	auto t_start = std::chrono::high_resolution_clock::now();

    cl_int status;

    cl_device_id device;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkOpenCLErrorCode("clGetDeviceIDs", status);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);

    // cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &status);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &status);
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    checkOpenCLErrorCode("clBuildProgram", status);
    if (status == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Program build log:\n");
        printf("%s\n", log);
    }

    cl_kernel kernel = clCreateKernel(program, "test_kernel", &status);
    checkOpenCLErrorCode("clCreateKernel", status);

    cl_mem parmN = 0;
    parmN = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &N, &status);
    checkOpenCLErrorCode("clCreateBuffer for parmN", status);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &parmN);
    checkOpenCLErrorCode("clSetKernelArg", status);

    // size_t localSize[1], globalSize[1];
    // // Number of work-items in each local work group, must be divisible by globalSize
    // localSize[0] = 1;
    // // Total number of work-items
    // globalSize[0] = N;

    int dimensions = 2;
    // Number of work-items in each local work group, must be divisible by globalSize
    size_t localSize[2] = { 2, 2 };
    // size_t localSize[1] = { 2*2 };
    // Total number of work-items
    size_t globalSize[2] = { 4, 4 };
    // size_t globalSize[1] = { 4*4 };

	printf("Launching OpenCL Kernel\n");
    if (dimensions == 1) {
        printf("Will run %lu threads, on a total of %lu threads in blocks\n", localSize[0], globalSize[0]);
    } else if (dimensions == 2) {
        printf("Will run %lu x %lu threads, on a total of %lu x %lu threads in blocks\n", localSize[0], localSize[1], globalSize[0], globalSize[1]);
    }

    status = clEnqueueNDRangeKernel(queue, kernel, dimensions, NULL, globalSize, localSize, 0, NULL, NULL);
    checkOpenCLErrorCode("clEnqueueNDRangeKernel", status);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
	

	auto t_end = std::chrono::high_resolution_clock::now();
	
	printf("Finishing OpenCL test, N = %d\t%ldms\n", N, std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    
    delete platforms;
}

