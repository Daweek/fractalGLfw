################################################################################
#Makefile to Generate Fractal test 	Edg@r J 2021 :)
################################################################################
#Compilers
GCC				= gcc
CXX 			= g++ -fPIC -std=c++17

CUDA 			= /usr/local/cuda-11.2
CUDA_SDK	= $(CUDA)/samples
NVCC     	= $(CUDA)/bin/nvcc

#Include Paths
CUDAINC   = -I. -I$(CUDA)/include -I$(CUDA_SDK)/common/inc -I/usr/include/x86_64-linux-gnu/qt5/
INC = -I$(HOME)/graphics/NumCpp/NumCpp/include
#Library Paths
CUDALIB		= -L/usr/lib/x86_64-linux-gnu -L$(CUDA)/lib64 \
						-lcuda -lcudart -lcudadevrt
GLLIB  		= -lGL -lGLU -lGLEW -lglfw
LIB 			= $(CUDALIB) $(GLLIB) -lQt5Core -lfreeimage -lstb -lm -lstdc++fs

################ Choosing architecture code for GPU ############################
NVCC_ARCH			=
HOSTNAME		 	= $(shell uname -n)

ifeq ("$(HOSTNAME)","Edgar-PC")
	NVCC_ARCH		= -gencode arch=compute_61,code=sm_61
endif

###############	Debug, 0 -> False,  1-> True
DEBUGON						:= 0

ifeq (1,$(DEBUGON))
	CXXDEBUG 				:= -g -Wall
	CXXOPT					:= -O0
#	NVCCDEBUG				:= -g -pg -G
	NVCCDEBUG				:= 
	NVCCOPT					:= -O0
	NVCCFLAGSXCOMP 	:= -Xcompiler -g,-Wall,-O0	
else
	CXXDEBUG 				:= 
	CXXOPT					:= -O3 -ffast-math -funroll-loops
	NVCCDEBUG				:= 
	NVCCOPT					:= -O3 --cudart=shared -use_fast_math
	NVCCFLAGSXCOMP 	:= -Xcompiler -O3,-ffast-math,-funroll-loops
endif
###############################################################################
CXXFLAGS				= $(CXXDEBUG) $(CXXOPT) -fopenmp
NVCCFLAGS 			= $(NVCCDEBUG) $(NVCC_DP) --compile $(NVCCOPT) $(NVCC_ARCH)
NVCCFLAGSLINK		= $(NVCCDEBUG) $(NVCC_DP) $(NVCCOPT) $(NVCC_ARCH)
###############################################################################

TARGET = frac

all: $(TARGET)

OBJLIST = shader.o text2D.o texture.o cuda.o

frac : main.o $(OBJLIST)
	$(NVCC) $(NVCCFLAGSLINK) $(NVCCFLAGSXCOMP) $(CUDAINC) $< -o $@ $(OBJLIST) $(LIB) 

main.o: main.cpp main.hpp
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@ 

cuda.o : cuda.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGSXCOMP) $(CUDAINC) $< -o $@ 
	
text2D.o: text2D.cpp text2D.hpp  
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@
	
shader.o: shader.cpp shader.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@	
	
texture.o: texture.cpp texture.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@	

clean:
	-rm -f *.o 
	-rm -f $(TARGET)
