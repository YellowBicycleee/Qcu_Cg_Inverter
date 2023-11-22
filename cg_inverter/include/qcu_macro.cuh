#pragma once

#include <stdio.h>
// #define DEBUG
#define N 6
#define RED "\033[31m"
#define BLUE "\e[0;34m" 
#define CLR "\033[0m"
#define L_RED                 "\e[1;31m"  
#define Nc 3
#define Nd 4
#define Ns 4


#define X_DIRECTION 0
#define Y_DIRECTION 1
#define Z_DIRECTION 2
#define T_DIRECTION 3

#define FRONT 1
#define BACK 0

// #define BLOCK_SIZE 128
#define BLOCK_SIZE 256
#define MAX_BLOCK_SIZE 256
#define WARP_SIZE 32


#define checkCudaErrors(err) \
  {                                                                                                                   \
    if (err != cudaSuccess) {                                                                                         \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                    \
              cudaGetErrorString(err), __FILE__, __LINE__);                                                           \
      exit(-1);                                                                                                       \
    }                                                                                                                 \
  }


__forceinline__ void qcuCudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t err = cudaMemcpy(dst, src, count, kind);
  if (err != cudaSuccess) {
    fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
             err, cudaGetErrorString(err), __FILE__, __LINE__);
    exit(-1);
  }
}

__forceinline__ void qcuCudaMalloc(void** devPtr, size_t size) {
  // cudaError_t cudaMalloc (void **devPtr, size_t  size );
  cudaError_t err = cudaMalloc(devPtr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
             err, cudaGetErrorString(err), __FILE__, __LINE__);
    exit(-1);
  }
}

// __forceinline__ void qcuCudaFree(void* ptr) {
//   cudaError_t err = cudaFree(ptr);
//   if (err != cudaSuccess) {
//     fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
//              err, cudaGetErrorString(err), __FILE__, __LINE__);
//     exit(-1);
//   }
// }


// __forceinline__ void qcuCudaDeviceSynchronize() {
//   cudaError_t err = cudaDeviceSynchronize();
//   if (err != cudaSuccess) {
//     fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
//              err, cudaGetErrorString(err), __FILE__, __LINE__);
//     exit(-1);
//   }
// }


#define qcuCudaMalloc(devPtr, size) {   \
  do {  \
    cudaError_t err = cudaMalloc(devPtr, size); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
              err, cudaGetErrorString(err), __FILE__, __LINE__);  \
      exit(-1); \
    } \
  } while(0); \
}


#define qcuCudaFree(ptr) {  \
  do {  \
    cudaError_t err = cudaFree(ptr);  \
    if (err != cudaSuccess) { \
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
              err, cudaGetErrorString(err), __FILE__, __LINE__);  \
      exit(-1); \
    } \
  } while (0);\
}



#define qcuCudaDeviceSynchronize() {           \
  do {                                          \
    cudaError_t err = cudaDeviceSynchronize();\
    if (err != cudaSuccess) {\
      fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
              err, cudaGetErrorString(err), __FILE__, __LINE__);\
      exit(-1);\
    } \
  } while(0); \
}
