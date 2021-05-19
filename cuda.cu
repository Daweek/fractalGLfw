#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// ***** CUDA includes
#include <cuda.h>
//#include <nvcuvid.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#define MAX_POINTS 5000000

// Globals for graphics
extern GLuint g_mapVBO;
struct cudaGraphicsResource* g_structMapVBO;
// For CUDA handlers
float	 *d_glmap;
extern float kernel_mili;
extern int numBlocks;
extern int blockSize;
extern int numMappings;
extern int numPoints;

typedef struct mp
{
  float x, y; // translation vertex
  float a, b, c, d; // scaling/rotation matrix
  float p; // mapping probability
} mapping;

mapping   *d_map;

extern "C"
void malloCUDA(mapping *mapped){
  std::cout<<"Malloc time...\n";
  // Prepare graphics interoperability
  if(g_structMapVBO != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(g_structMapVBO));

  glDeleteBuffers(1,&g_mapVBO);

  // Creation of share buffer between CUDA and OpenGL
  // For mapping position and color
  glGenBuffers(1, &g_mapVBO);
  glBindBuffer(GL_ARRAY_BUFFER, g_mapVBO);
  unsigned int size = MAX_POINTS * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Register CUDA and OpenGL Interop
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_structMapVBO,g_mapVBO,cudaGraphicsMapFlagsNone));

  // Send all parameters to GPU
  checkCudaErrors(cudaFree(d_map));
  checkCudaErrors(cudaMalloc((void**)&d_map,5*sizeof(mapping)));
  checkCudaErrors(cudaMemcpy(d_map,mapped,5*sizeof(mapping),cudaMemcpyHostToDevice));

}

__global__ void kernel(float4* d_pointData, int numPoints,mapping *d_mappings, int numMappings)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // If needed for performance, move curand_init to seperate kernel and store
  // states in device memory
  curandState state;
  curand_init((unsigned long long) clock(), index, 0, &state);

  // Set up transformation mapping once per block in shared memory
  extern __shared__ mapping maps[];
  if(threadIdx.x == 0)
  {
    for(int i = 0; i < numMappings; i++)
        maps[i] = d_mappings[i];
  }
  __syncthreads();

  // Initially start at a mapping vertex to guarantee we stay inside the
  // iterated function system
  int currentTarget = index % numMappings;
  float2 currentPosition, newPosition;
  currentPosition.x = maps[currentTarget].x;
  currentPosition.y = maps[currentTarget].y;

  for(int i = index; i < numPoints; i += stride)
  {
    // set the current vertex to the currentPosition
    d_pointData[i].x = currentPosition.x;
    d_pointData[i].y = currentPosition.y;

    // set the iteration percentage and current target mapping
    d_pointData[i].z =  i / (float) numPoints;
    d_pointData[i].w = currentTarget;

    // find random target with given mapping probabilities
    // If needed for performance, find method to remove thread divergence
    // Note: changing 4 to numMappings in for loop reduced performance 50%
    float currentProb = curand_uniform(&state);
    float totalProb = 0.0f;
    for(int j = 0; j < numMappings; j++)
    {
        totalProb += maps[j].p;
        if(currentProb < totalProb)
        {
            currentTarget = j;
            break;
        }
    }

    // calculate the transformation
    // (x_n+1) = (a b)(x_n) + (e)
    // (y_n+1)   (c d)(y_n)   (f)
    newPosition.x = maps[currentTarget].a * currentPosition.x +
                    maps[currentTarget].b * currentPosition.y +
                    maps[currentTarget].x;
    newPosition.y = maps[currentTarget].c * currentPosition.x +
                    maps[currentTarget].d * currentPosition.y +
                    maps[currentTarget].y;
    currentPosition = newPosition;
  }
}

extern "C"
void generateFractal(){
  
  size_t mapsizevbo;
  checkCudaErrors(cudaGraphicsMapResources(1,&g_structMapVBO,0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glmap,&mapsizevbo,g_structMapVBO));
  
  cudaEvent_t start, stop;
  checkCudaErrors( cudaEventCreate(&start) );
  checkCudaErrors( cudaEventCreate(&stop) );
  
  checkCudaErrors( cudaEventRecord(start) );
  
    kernel<<<numBlocks, blockSize, numMappings * sizeof(mapping)>>>
      ((float4*)d_glmap, numPoints, d_map, numMappings);
  
  checkCudaErrors( cudaEventRecord(stop) );

  // handle any synchronous and asynchronous kernel errors
  checkCudaErrors( cudaGetLastError() );
  checkCudaErrors( cudaDeviceSynchronize() );

  // record and print kernel timing
  checkCudaErrors( cudaEventSynchronize(stop) );
  kernel_mili = 0;
  checkCudaErrors( cudaEventElapsedTime(&kernel_mili, start, stop) );

  // Unmap OpenGL resources
  checkCudaErrors(cudaGraphicsUnmapResources(1,&g_structMapVBO,0));

}

extern "C"
void freeCUDA(){
  // Unregister if CUDA-InteropGL
	printf("Unregistering Resources...\n");
  //checkCudaErrors(cudaGraphicsUnmapResources(1,&g_structMapVBO,0));
	//checkCudaErrors(cudaGraphicsUnregisterResource(g_structMapVBO));
  //Freeing CUDA
  std::cout<<"Freeing CUDA...\n";
  checkCudaErrors(cudaFree(d_map));

}