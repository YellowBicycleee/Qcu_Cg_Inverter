
#include "qcu_complex_computation.cuh"
#include "qcu_complex.cuh"
#include "qcu_macro.cuh"
#include <cuda_runtime.h>

// #define DEBUG

// DESCRIBE：  x = ax
static __global__ void sclar_multiply_vector_gpu (void* x, void* a, int vol) {
  // a : complex
  int pos = blockIdx.x * blockDim.x + threadIdx.x;  // thread pos in total
  Complex scalar = *(static_cast<Complex*>(a));

  Complex* x_dst = static_cast<Complex*>(x) + pos * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    x_dst[i] = scalar * x_dst[i];
  }
}

// FUNCTION: saxpy_gpu
// DESCRIBE：  y = ax + y，vol is Lx * Ly * Lz * Lt
static __global__ void saxpy_gpu (void* y, void* x, void* a, int vol) {
  // a : complex
  int pos = blockIdx.x * blockDim.x + threadIdx.x;  // thread pos in total
  Complex scalar = *(static_cast<Complex*>(a));

  Complex* y_dst = static_cast<Complex*>(y) + pos * Ns * Nc;
  Complex* x_dst = static_cast<Complex*>(x) + pos * Ns * Nc;

// #ifdef DEBUG
//   if (pos == 0) {
//     printf("before real = %lf, imag = %lf\nafter real = %lf imag = %lf\n", y_dst[0].real(), y_dst[0].imag(), (y_dst[0] + x_dst[0] * scalar).real(), (y_dst[0] + x_dst[0] * scalar).imag());
//   }
// #endif

  for (int i = 0; i < Ns * Nc; i++) {
    y_dst[i] += x_dst[i] * scalar;
  }
}

// FUNCTION: inner product
static __global__ void partial_product_kernel(void* x, void* y, void* partial_result, int vol) {
  int thread_in_total = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_in_block = threadIdx.x;

  int stride, last_stride;
  Complex* x_ptr = static_cast<Complex*>(x);
  Complex* y_ptr = static_cast<Complex*>(y);

// if(thread_in_total == 0) {
//   printf("x_ptr = %p, y_ptr = %p\n", x_ptr, y_ptr);
// }

  Complex temp;
  temp.clear2Zero();
  __shared__ Complex cache[BLOCK_SIZE];

  for (int i = thread_in_total; i < vol * Ns * Nc; i += vol) {
    temp += x_ptr[i] * (y_ptr[i].conj());
    // temp += x_ptr[i] * x_ptr[i].conj();
  }
  cache[thread_in_block] = temp;
  __syncthreads();
  // reduce in block
  last_stride = BLOCK_SIZE;
  stride = BLOCK_SIZE / 2;
  while (stride > 0 && thread_in_block < stride) {
    if (thread_in_block + stride < last_stride) {
      cache[thread_in_block] += cache[thread_in_block + stride];
    }
    stride /= 2;
    last_stride /= 2;
    __syncthreads();
  }
  if (thread_in_block == 0) {
    *(static_cast<Complex*>(partial_result) + blockIdx.x) = cache[0];
  }
}



// when call this function, set gridDim to 1
static __global__ void reduce_partial_result(void* partial_result, int partial_length) {
  int thread_in_block = threadIdx.x;
  int stride, last_stride;

  Complex temp;
  Complex* src = static_cast<Complex*>(partial_result);
  temp.clear2Zero();
  __shared__ Complex cache[BLOCK_SIZE];

  for (int i = thread_in_block; i < partial_length; i+= BLOCK_SIZE) {
    temp += src[i];
  }
  cache[thread_in_block] = temp;
  __syncthreads();
  // reduce in block
  last_stride = BLOCK_SIZE;
  stride = BLOCK_SIZE / 2;
  while (stride > 0 && thread_in_block < stride) {
    // if (thread_in_block < stride && thread_in_block + stride < last_stride) {
    if (thread_in_block + stride < last_stride) {
      cache[thread_in_block] += cache[thread_in_block + stride];
    }
    stride /= 2;
    last_stride /= 2;
    __syncthreads();
  }
  if (thread_in_block == 0) {
    *(static_cast<Complex*>(partial_result) + thread_in_block) = cache[0];
  }
}


// avoid malloc partial_result
void gpu_inner_product (void* x, void* y, void* result, void* partial_result, int vol) {
  int grid_size = vol / BLOCK_SIZE;
  int block_size = BLOCK_SIZE;

  partial_product_kernel<<<grid_size, block_size>>>(x, y, partial_result, vol);
  checkCudaErrors(cudaDeviceSynchronize());
  reduce_partial_result<<<1, block_size>>>(partial_result, grid_size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(result, partial_result, sizeof(Complex), cudaMemcpyDeviceToDevice));
}

void gpu_sclar_multiply_vector (void* x, void* scalar, int vol) {
  dim3 gridDim(vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  sclar_multiply_vector_gpu<<<gridDim, blockDim>>>(x, scalar, vol);
  checkCudaErrors(cudaDeviceSynchronize());
}


// every point has Ns * Nc dim vector, y <- y + scalar * x
void gpu_saxpy(void* x, void* y, void* scalar, int vol) {
  dim3 gridDim(vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  saxpy_gpu<<<gridDim, blockDim>>>(y, x, scalar, vol);
  checkCudaErrors(cudaDeviceSynchronize());
}





__global__ void norm2_gpu (void* vector, void* partial_result, int vector_length, void* result){
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  int thread_in_block = threadIdx.x;
  int vol = gridDim.x * blockDim.x;
  double temp = 0;
  double temp_res = 0;
  int stride;
  int last_stride;

  __shared__ double cache_res[MAX_BLOCK_SIZE];

  if (thread_id < vector_length) {
    Complex *start_ptr = static_cast<Complex*>(vector);
    temp = 0;
    temp_res = 0;
    for (int i = thread_id; i < vector_length; i += vol) {
      temp = start_ptr[i].norm2();
      temp_res += temp * temp;
    }

    // reduce
    cache_res[thread_in_block] = temp_res;
    __syncthreads();
    last_stride = blockDim.x;
    stride = last_stride / 2;
    while (stride > 0 && thread_in_block < stride) {
      if (thread_in_block + stride < last_stride) {
        cache_res[thread_in_block] += cache_res[thread_in_block + stride];
        last_stride /= 2;
        stride /= 2;
      }
      __syncthreads();
    }

    // store
    if (thread_in_block == 0) {
      *(static_cast<double*>(partial_result) + blockIdx.x) = cache_res[0];
    }
  }
}

__global__ void reduce_norm2_gpu(void* partial_result, int partial_length) {
  int thread_in_block = threadIdx.x;
  int stride, last_stride;

  double temp = 0;
  double* src = static_cast<double*>(partial_result);

  __shared__ double cache[MAX_BLOCK_SIZE];

  for (int i = thread_in_block; i < partial_length; i+= BLOCK_SIZE) {
    temp += src[i];
  }
  cache[thread_in_block] = temp;
  __syncthreads();
  // reduce in block
  last_stride = blockDim.x;
  stride = last_stride / 2;
  while (stride > 0 && thread_in_block < stride) {
    if (thread_in_block + stride < last_stride) {
      cache[thread_in_block] += cache[thread_in_block + stride];
    }
    stride /= 2;
    last_stride /= 2;
    __syncthreads();
  }
  if (thread_in_block == 0) {
    *(static_cast<double*>(partial_result) + thread_in_block) = cache[0];
  }
}


void gpu_vector_norm2(void* vector, void* temp_res, int vector_length, void* result) {
  int block_size = MAX_BLOCK_SIZE;
  int grid_size = (vector_length + block_size - 1 ) / block_size;

  norm2_gpu<<<grid_size, block_size>>> (vector, temp_res, vector_length, result);
  checkCudaErrors(cudaDeviceSynchronize());
  // reduce
  reduce_norm2_gpu<<<1, block_size>>>(temp_res, grid_size);
  checkCudaErrors(cudaDeviceSynchronize());
  double square_norm2;
  double res;
  checkCudaErrors(cudaMemcpy(&square_norm2, temp_res, sizeof(double), cudaMemcpyDeviceToHost));
  res = sqrt(square_norm2);
  checkCudaErrors(cudaMemcpy(result, &res, sizeof(double), cudaMemcpyHostToDevice));

#ifdef DEBUG
  printf(RED"file <%s>, line <%d>, function <%s>, norm2 in process = %lf\n", __FILE__, __LINE__, __FUNCTION__, res);
  printf(CLR"");
#endif

  // Complex* host_vector1;
  // double square_norm1 = 0;
  // host_vector1 = new Complex[vector_length];
  // // checkCudaErrors(cudaMalloc(&host_vector1, sizeof(Complex) * vector_length));
  // checkCudaErrors(cudaMemcpy(host_vector1, vector, sizeof(Complex) * vector_length, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vector_length; i++) {
  //   square_norm1 += host_vector1[i].norm2() * host_vector1[i].norm2();
  // }
  // double res1 = sqrt(square_norm1);
  // checkCudaErrors(cudaMemcpy(result, &res1, sizeof(double), cudaMemcpyHostToDevice));
  // printf("gpu_norm = %.32lf, cpu_norm = %.32lf, diff = %g\n", res, res1, res - res1);
  // delete []host_vector1;
}




// FUNCTION: inner product
static __global__ void partial_inner_prod_kernel(void* x, void* y, \
    void* partial_result, int vector_length
) {
  int vol = gridDim.x * blockDim.x;
  int thread_in_total = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_in_block = threadIdx.x;
  int stride, last_stride;
  int block_size = blockDim.x;
  Complex temp;
  Complex* x_ptr = static_cast<Complex*>(x);
  Complex* y_ptr = static_cast<Complex*>(y);

  if (thread_in_total >= vol) {
    return;
  }

  temp.clear2Zero();
  __shared__ Complex cache[BLOCK_SIZE];

  for (int i = thread_in_total; i < vector_length; i += vol) {
    temp += x_ptr[i] * y_ptr[i].conj();
  }
  cache[thread_in_block] = temp;
  __syncthreads();
  // reduce in block
  last_stride = block_size;
  stride = block_size / 2;
  while (stride > 0 && thread_in_block < stride) {
    if (thread_in_block + stride < last_stride) {
      cache[thread_in_block] += cache[thread_in_block + stride];
    }
    stride /= 2;
    last_stride /= 2;
    __syncthreads();
  }
  if (thread_in_block == 0) {
    *(static_cast<Complex*>(partial_result) + blockIdx.x) = cache[0];
  }
}

void gpu_inner_product_new (void* x, void* y, void* result, void* partial_result, int vector_length) {
  int grid_size = (vector_length + BLOCK_SIZE * Ns * Nc -1) / (BLOCK_SIZE * Ns * Nc);
  int block_size = BLOCK_SIZE;

  partial_inner_prod_kernel<<<grid_size, block_size>>>(x, y, partial_result, vector_length);
  checkCudaErrors(cudaDeviceSynchronize());
  reduce_partial_result<<<1, block_size>>>(partial_result, grid_size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(result, partial_result, sizeof(Complex), cudaMemcpyDeviceToDevice));
}