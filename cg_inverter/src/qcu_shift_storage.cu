#include "qcu_shift_storage.cuh"
#include <cstdio>
#include "qcu_macro.cuh"
// TODO: WARP version, no sync  
static __device__ void loadVectorBySharedMemory(void* origin, void* result) {
  __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;

  // result is register variable
  double* shared_dst = shared_buffer + threadIdx.x * Ns * Nc * 2;
  double* warp_src = static_cast<double*>(origin) + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc * 2;

  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i] = warp_src[i];
  }

  // load data to register
  double* register_addr = static_cast<double*>(result);
  for (int i = 0; i < Ns * Nc * 2; i++) {
    register_addr[i] = shared_dst[i];
  }
}


// Lx is full Lx, not Lx / 2
static __global__ void shift_vector_to_coalesed (void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt) {
  // change storage to [parity, Ns, Nc, 2, t, z, y, x]
  // __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];
  int sub_Lx = Lx >> 1;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  // double* src_vec_pointer = static_cast<double*>(src_vec) + thread_id * Ns * Nc * 2;
  double* dst_vec_pointer = static_cast<double*>(dst_vec) + thread_id;

  // mofify 
  double data_local[Ns * Nc * 2];
  loadVectorBySharedMemory(src_vec, data_local);

  for (int i = 0; i < Ns * Nc * 2; i++) {
    *dst_vec_pointer = data_local[i];
    dst_vec_pointer += sub_vol;
  }

  /*
  for (int i = 0; i < Ns * Nc * 2; i++) {
    *dst_vec_pointer = src_vec_pointer[i];
    dst_vec_pointer += sub_vol;
  }*/
  
}

// Lx is full Lx, not Lx / 2
static __global__ void shift_vector_to_noncoalesed (void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt) {
  // change storage to [parity, Ns, Nc, 2, t, z, y, x]
  int sub_Lx = Lx >> 1;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  double* dst_vec_pointer = static_cast<double*>(src_vec) + thread_id * Ns * Nc * 2;
  double* src_vec_pointer = static_cast<double*>(dst_vec) + thread_id;

  for (int i = 0; i < Ns * Nc * 2; i++) {
    dst_vec_pointer[i] = *src_vec_pointer;
    src_vec_pointer += sub_vol;
  }
}


// Lx is full Lx, not Lx / 2
static __global__ void shift_gauge_to_coalesed (void* dst_gauge, void* src_gauge, int Lx, int Ly, int Lz, int Lt) {
  // change storage to [parity, Ns, Nc, 2, t, z, y, x]
  int sub_Lx = Lx >> 1;
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  double* dst_gauge_ptr;
  double* src_gauge_ptr;
  for (int i = 0; i < Nd; i++) {
    dst_gauge_ptr = static_cast<double*>(dst_gauge) + 2 * sub_vol * Nc * Nc;
    src_gauge_ptr = static_cast<double*>(src_gauge) + 2 * sub_vol * Nc * Nc;

    for (int i = 0 ; i < Nc * (Nc-1) * 2; i++) {
      dst_gauge_ptr[i * Nc * (Nc - 1) + thread_id] = src_gauge_ptr[thread_id * Nc * Nc + i];
    }
  }
}




void shiftVectorStorage(void* dst_vec, void* src_vec, int shift_direction, int Lx, int Ly, int Lz, int Lt) {
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;

  int block_size = 128;
  int grid_size = (half_vol + block_size - 1) / block_size;

  if (shift_direction == TO_COALESCE) {
    shift_vector_to_coalesed <<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    checkCudaErrors(cudaDeviceSynchronize());
  } else {
    shift_vector_to_noncoalesed <<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    checkCudaErrors(cudaDeviceSynchronize());
  }
}


void shiftGaugeStorage(void* dst_vec, void* src_vec, int shift_direction, int Lx, int Ly, int Lz, int Lt) {
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol / 2;

  int block_size = 256;
  int grid_size = (half_vol + block_size - 1) / block_size;

  if (shift_direction == TO_COALESCE) {
    shift_gauge_to_coalesed <<<grid_size, block_size>>>(dst_vec, src_vec, Lx, Ly, Lz, Lt);
    checkCudaErrors(cudaDeviceSynchronize());
  }
}