# Author: Dinei Rockenbach <dinei.rockenbach@edu.pucrs.br> 
# Date:   February 2019
# Last update: 02/may/2019
# Description: this is the make file for compiling all programs 
#
# Define MANDEL_NO_DISPLAY if you do not want to display image
# ex: export MANDEL_NO_DISPLAY=1  or
#     setenv MANDEL_NO_DISPLAY 1  or 
# 	  make MANDEL_NO_DISPLAY=1 -j10


ifdef MANDEL_NO_DISPLAY
	DIS +=-DNO_DISPLAY
endif
ifdef MANDEL_NO_GPU
	DIS +=-DNO_GPU
endif
DIS					+= -DNODBG -DNO_CMAKE_CONFIG

IFLAGS				= 
LFLAGS				= 
LIBS				= 
ifndef MANDEL_NO_DISPLAY
	IFLAGS			= -Ilibs
	LFLAGS			= -Llibs
	LIBS			= -lmarX2 -lX11 -lm
endif

CPP                 = g++
CPPFLAGS            = $(DIS) $(IFLAGS) $(LFLAGS) -std=c++14 -O3 -Wall -Wno-aligned-new

SPAR              	= $(HOME)/SPar/bin/spar
SPARFLAGS           = $(CPPFLAGS)

FFPATH				= $(HOME)/SPar/libraries
FF					= $(CPP)
FFONLYFLAGS			= -I$(FFPATH)
FFFLAGS				= $(CPPFLAGS) $(FFONLYFLAGS)

TBB					= $(CPP)
TBBFLAGS			= $(CPPFLAGS)

NVCC				= nvcc
NVCCFLAGS			= $(DIS) $(IFLAGS) $(LFLAGS) -std=c++14 -O3 -x cu -D_FORCE_INLINES --default-stream per-thread

OCL 				= $(CPP)
OCLFLAGS			= $(DIS) $(IFLAGS) $(LFLAGS) -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -std=c++14 -O3

IFLAGS				= 
LFLAGS				= 
LIBS				= 
ifndef MANDEL_NO_DISPLAY
	IFLAGS			= -Ilibs
	LFLAGS			= -Llibs
	LIBS			= -lmarX2 -lX11 -lm
endif


all: mandel spar fastflow tbb cuda opencl sparcuda sparopencl tbbcuda tbbopencl fastflowcuda fastflowopencl

test:
	$(NVCC) $(NVCCFLAGS) cuda_test.cu -o bin_cuda_test $(LIBS)
	$(OCL) $(OCLFLAGS) opencl_test.cpp -o bin_opencl_test $(LIBS) -lOpenCL

# Sequential
mandel: mandel.cpp
	$(CPP) $(CPPFLAGS) $< -o bin_seq_$@ $(LIBS)

# CPU

spar: spar_mandel.cpp
	$(SPAR) $(SPARFLAGS) -spar_file $< -o bin_$@ $(LIBS) -pthread

fastflow: fastflow_mandel.cpp
	$(FF) $(FFFLAGS) $< -o bin_$@ $(LIBS) -pthread

tbb: tbb_mandel.cpp
	$(TBB) $(TBBFLAGS) $< -o bin_$@ $(LIBS) -ltbb

# GPU

cuda: cuda_mandel.cu
	$(NVCC) $(NVCCFLAGS) $< -o bin_$@ $(LIBS)

opencl: opencl_mandel.cpp
	$(OCL) $(OCLFLAGS) $< -o bin_$@ $(LIBS) -lOpenCL

# CPU + GPU

sparcuda: sparcuda_mandel.cpp
	$(NVCC) $(NVCCFLAGS) $(FFONLYFLAGS) $< -o bin_$@ $(LIBS)

sparopencl: sparopencl_mandel.cpp
	$(OCL) $(OCLFLAGS) $(FFONLYFLAGS) $< -o bin_$@ $(LIBS) -pthread -lOpenCL

tbbcuda: tbbcuda_mandel.cu
	$(NVCC) $(NVCCFLAGS) $< -o bin_$@ $(LIBS) -ltbb

tbbopencl: tbbopencl_mandel.cpp
	$(OCL) $(OCLFLAGS) $(FFONLYFLAGS) $< -o bin_$@ $(LIBS) -ltbb -lOpenCL

fastflowcuda: fastflowcuda_mandel.cu
	$(NVCC) $(NVCCFLAGS) $(FFONLYFLAGS) $< -o bin_$@ $(LIBS)

fastflowopencl: fastflowopencl_mandel.cpp
	$(OCL) $(OCLFLAGS) $(FFONLYFLAGS) $< -o bin_$@ $(LIBS) -pthread -lOpenCL


clean:
	rm -f bin_*
