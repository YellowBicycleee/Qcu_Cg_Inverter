#include "qcu_wilson_dslash_neo.cuh"
#include "qcu_complex.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
#include "qcu_shift_storage_complex.cuh"
#include <chrono>
#define INCLUDE_COMPUTATION

extern int grid_x;
extern int grid_y;
extern int grid_z;
extern int grid_t;
extern MPICommunicator *mpi_comm;

static void* coalesced_fermion_in;
static void* coalesced_fermion_out;
// static void* coalesced_gauge;  ---->qcu_gauge
bool memory_allocated;


extern void* qcu_gauge;


// WARP version, no sync
// static __device__ __forceinline__ void storeVectorBySharedMemory(void* shared_ptr, Complex* origin, Complex* result) {
//   // result is register variable
//   double* shared_buffer = static_cast<double*>(shared_ptr);
//   int thread = blockDim.x * blockIdx.x + threadIdx.x;
//   int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;
//   Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;
//   Complex* warp_dst = origin + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc;
//   double* double_dst = reinterpret_cast<double*>(warp_dst);

//   // store result to shared memory
//   for (int i = 0; i < Ns * Nc; i++) {
//     shared_dst[i] = result[i];
//   }
//   // store result of shared memory to global memory
//   for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
//     double_dst[i] = shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i];
//   }
// }



static __device__ __forceinline__ void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}

__device__ __forceinline__ void loadGauge(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* u = p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
__device__ __forceinline__ void loadVector(Complex* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}



static __device__ __forceinline__ void loadGaugeCoalesced(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int sub_Lx, int Ly, int Lz, int Lt) {
  Complex* start_ptr = p.getCoalescedGaugeAddr (gauge_ptr, direction, sub_Lx, Ly, Lz, Lt);
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = *start_ptr;
    start_ptr += sub_vol;
  }
  reconstructSU3(u_local);
}

static __device__ __forceinline__ void loadVectorCoalesced(Complex* src_local, void* fermion_in, const Point& p, int half_Lx, int Ly, int Lz, int Lt) {
  Complex* start_ptr = p.getCoalescedVectorAddr (fermion_in, half_Lx, Ly, Lz, Lt);
  int sub_vol = half_Lx * Ly * Lz * Lt;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = *start_ptr;
    start_ptr += sub_vol;
  }
}

static __device__ __forceinline__ void storeVectorCoalesced(Complex* dst_local, void* fermion_out, const Point& p, int half_Lx, int Ly, int Lz, int Lt) {
  Complex* start_ptr = p.getCoalescedVectorAddr (fermion_out, half_Lx, Ly, Lz, Lt);
  int sub_vol = half_Lx * Ly * Lz * Lt;

  for (int i = 0; i < Ns * Nc; i++) {
    *start_ptr = dst_local[i];
    start_ptr += sub_vol;
  }
}


static __global__ void wilsonDslashFull(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, double kappa) {
  int half_Lx = Lx >> 1;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread_id / (Lz * Ly * half_Lx);
  int z = thread_id % (Lz * Ly * half_Lx) / (Ly * half_Lx);
  int y = thread_id % (Ly * half_Lx) / half_Lx;
  int x = thread_id % half_Lx;

  Point p(x, y, z, t, parity);
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  loadVectorCoalesced(src_local, fermion_in, p, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);

  // src - kappa dst
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] = src_local[i] - dst_local[i] * kappa;
  }
  storeVectorCoalesced(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);  // store result
}


static __global__ void mpiDslashCoalesce(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, double flag_param) {
  assert(parity == 0 || parity == 1);
  Lx >>= 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread_id / (Lz * Ly * Lx);
  int z = thread_id % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread_id % (Ly * Lx) / Lx;
  int x = thread_id % Lx;

  int coord_boundary;
  double flag = flag_param;


  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex temp1;
  Complex temp2;
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  // loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, X_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multipy_i() * flag;
    }
#endif
  }

  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, X_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj

        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multipy_minus_i() * flag;
    }
#endif
  }


  // \mu = 2
  // y front
  loadGaugeCoalesced(u_local, gauge, Y_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j] *  flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += -temp2 * flag;
    }
#endif
  }

  // y back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, Y_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);


  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += -temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2 * flag;
    }
#endif
  }

  // \mu = 3
  // z front
  loadGaugeCoalesced(u_local, gauge, Z_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {
#ifdef INCLUDE_COMPUTATION

#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multipy_minus_i() * flag;
    }
#endif
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, Z_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multipy_i() * flag;
    }
#endif
  }

  // t: front
  // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, T_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j] * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += -temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += -temp2 * flag;
    }
#endif
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, T_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2 * flag;
    }
#endif
  }

  // store result
  storeVectorCoalesced(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);
}



void WilsonDslash::calculateDslash(int invert_flag) {
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int parity = dslashParam_->parity;
  double flag;
  if (invert_flag == 0) {
    flag = 1.0;
  } else {
    flag = -1.0;
  }


  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);


  mpi_comm->preDslash(dslashParam_->fermion_in, parity, invert_flag);

  auto start = std::chrono::high_resolution_clock::now();
  void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t, &flag};


  checkCudaErrors(cudaLaunchKernel((void *)mpiDslashCoalesce, gridDim, blockDim, args));

  checkCudaErrors(cudaDeviceSynchronize());
  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("coalescing total time: (without malloc free memcpy) : %.9lf sec, block size %d\n", double(duration) / 1e9, BLOCK_SIZE);

}

void WilsonDslash::calculateDslashFull(double kappa, int dagger_flag) {
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int parity = dslashParam_->parity;
  double flag;
  if (dagger_flag == 0) {
    flag = 1.0;
  } else {
    flag = -1.0;
  }

  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  mpi_comm->preDslash(dslashParam_->fermion_in, parity, dagger_flag);

  auto start = std::chrono::high_resolution_clock::now();
  void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t, &flag};


  checkCudaErrors(cudaLaunchKernel((void *)mpiDslashCoalesce, gridDim, blockDim, args));

  checkCudaErrors(cudaDeviceSynchronize());
  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, dagger_flag);

  // wilsonDslashFull(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, double kappa) 
  // kappa
  void *args1[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &kappa};
  checkCudaErrors(cudaLaunchKernel((void *)wilsonDslashFull, gridDim, blockDim, args1));


  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("coalescing total time, with kappa: (without malloc free memcpy) : %.9lf sec, block size %d\n", double(duration) / 1e9, BLOCK_SIZE);
}




// static __global__ void mpiDoNothing(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, double flag_param) {
// }





static __global__ void mpiDslashNaive(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, double flag_param) {
  assert(parity == 0 || parity == 1);

  Lx >>= 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread_id / (Lz * Ly * Lx);
  int z = thread_id % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread_id % (Ly * Lx) / Lx;
  int x = thread_id % Lx;

  int coord_boundary;
  double flag = flag_param;


  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  // Complex temp;
  Complex temp1;
  Complex temp2;
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGauge(u_local, gauge, X_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {

#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multipy_i() * flag;
    }
#endif
  }
  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);;

  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj

        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multipy_minus_i() * flag;
    }
#endif
  }

  // \mu = 2
  // y front
  loadGauge(u_local, gauge, Y_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j] *  flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += -temp2 * flag;
    }
#endif
  }

  // y back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);


  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += -temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2 * flag;
    }
#endif
  }

  // \mu = 3
  // z front
  loadGauge(u_local, gauge, Z_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multipy_minus_i() * flag;
    }
#endif
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multipy_i() * flag;
    }
#endif
  }

  // t: front
  // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j] * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += -temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += -temp2 * flag;
    }
#endif
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2 * flag;
    }
#endif
  }

  // store result
  // storeVectorCoalesced(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);
  // store result
  // storeVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), dst_local);

  Complex* dst_global = p.getPointVector(static_cast<Complex *>(fermion_out), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    dst_global[i] = dst_local[i];
  }
  
}




void WilsonDslash::calculateDslashNaive(int invert_flag) {
  int Lx = dslashParam_->Lx;
  int Ly = dslashParam_->Ly;
  int Lz = dslashParam_->Lz;
  int Lt = dslashParam_->Lt;
  int parity = dslashParam_->parity;
  double flag;
  if (invert_flag == 0) {
    flag = 1.0;
  } else {
    flag = -1.0;
  }

  int space = Lx * Ly * Lz * Lt >> 1;
  dim3 gridDim(space / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  checkCudaErrors(cudaDeviceSynchronize());

  mpi_comm->preDslash(dslashParam_->fermion_in, parity, invert_flag);

  auto start = std::chrono::high_resolution_clock::now();
  void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t, &flag};

  checkCudaErrors(cudaLaunchKernel((void *)mpiDslashNaive, gridDim, blockDim, args));

  checkCudaErrors(cudaDeviceSynchronize());
  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  // printf("naive total time: (without malloc free memcpy) : %.9lf sec, block size = %d\n", double(duration) / 1e9, BLOCK_SIZE);
}

void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;

  if (!memory_allocated) {
    checkCudaErrors(cudaMalloc(&coalesced_fermion_in, sizeof(double) * vol / 2 * Ns * Nc * 2));
    checkCudaErrors(cudaMalloc(&coalesced_fermion_out, sizeof(double) * vol / 2 * Ns * Nc * 2));
    memory_allocated = true;
  }

  shiftVectorStorageTwoDouble(coalesced_fermion_in, fermion_in, TO_COALESCE, Lx, Ly, Lz, Lt);

  DslashParam dslash_param(coalesced_fermion_in, coalesced_fermion_out, gauge, param, parity);
  WilsonDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash(invert_flag);

  shiftVectorStorageTwoDouble(fermion_out, coalesced_fermion_out, TO_NON_COALESCE, Lx, Ly, Lz, Lt);
}



void callWilsonDslashFull(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  double kappa = 1.0;
  if (!memory_allocated) {
    checkCudaErrors(cudaMalloc(&coalesced_fermion_in, sizeof(double) * vol / 2 * Ns * Nc * 2));
    checkCudaErrors(cudaMalloc(&coalesced_fermion_out, sizeof(double) * vol / 2 * Ns * Nc * 2));
    memory_allocated = true;
  }

  shiftVectorStorageTwoDouble(coalesced_fermion_in, fermion_in, TO_COALESCE, Lx, Ly, Lz, Lt);

  DslashParam dslash_param(coalesced_fermion_in, coalesced_fermion_out, qcu_gauge, param, parity);
  WilsonDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslashFull(kappa, 0);

  shiftVectorStorageTwoDouble(fermion_out, coalesced_fermion_out, TO_NON_COALESCE, Lx, Ly, Lz, Lt);
}


void callWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  WilsonDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslashNaive(invert_flag);
}





__attribute__((constructor)) void init_wilson () {

  coalesced_fermion_in = nullptr;
  coalesced_fermion_out = nullptr;
  memory_allocated = false;
}

__attribute__((destructor)) void destroy_wilson () {
  memory_allocated = false;
}