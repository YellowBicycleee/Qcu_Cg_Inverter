#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "qcu.h"
#include <cuda_runtime.h>
#include "qcu_complex.cuh"
#include "qcu_dslash.cuh"
#include "qcu_macro.cuh"
// #define DEBUG
// #define N 6
// #define RED "\033[31m"
// #define BLUE "\e[0;34m" 
// #define CLR "\033[0m"
// #define L_RED                 "\e[1;31m"  
// #define Nc 3
// #define Nd 4
// #define Ns 4
// // #define X_FRONT 1
// // #define X_BACK -1
// // #define Y_FRONT 2
// // #define Y_BACK -2
// // #define Z_FRONT 3
// // #define Z_BACK -3
// // #define T_FRONT 4
// // #define T_BACK -4

// #define X_DIRECTION 0
// #define Y_DIRECTION 1
// #define Z_DIRECTION 2
// #define T_DIRECTION 3

// #define FRONT 1
// #define BACK 0

// #define BLOCK_SIZE 128
// #define WARP_SIZE 32
#define qcuPrint() { \
    printf("function %s line %d...\n", __FUNCTION__, __LINE__); \
}

// #define checkCudaErrors(err)                                                                                          \
//   {                                                                                                                   \
//     if (err != cudaSuccess) {                                                                                         \
//       fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", err,                    \
//               cudaGetErrorString(err), __FILE__, __LINE__);                                                           \
//       exit(-1);                                                                                                       \
//     }                                                                                                                 \
//   }


// static / global variables
static int grid_x;
static int grid_y;
static int grid_z;
static int grid_t;

static int process_rank;
static int process_num;
// static cudaStream_t stream[Nd][2];  // Nd means dims, 2 means FRONT BACK
// end


struct Coord {  // use this to store the coord of this process, and calculate adjacent process
  int x;
  int y;
  int z;
  int t;
  Coord() = default;
  Coord(int p_x, int p_y, int p_z, int p_t) : x(p_x), y(p_y), z(p_z), t(p_t) {}
  int calculateMpiRank() const {
    return ((x*grid_y + y)*grid_z+z)*grid_t + t;
  }
  Coord adjCoord(int front_back, int direction) const {
    // suppose all of grid_x, grid_y, grid_z, grid_t >= 1
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);

    int new_pos;
    switch (direction) {
      case X_DIRECTION:
        new_pos = (front_back == FRONT) ? ((x+1)%grid_x) : ((x+grid_x-1)%grid_x);
        return Coord(new_pos, y, z, t);
        break;
      case Y_DIRECTION:
        new_pos = (front_back == FRONT) ? ((y+1)%grid_y) : ((y+grid_y-1)%grid_y);
        return Coord(x, new_pos, z, t);
        break;
      case Z_DIRECTION:
        new_pos = (front_back == FRONT) ? ((z+1)%grid_z) : ((z+grid_z-1)%grid_z);
        return Coord(x, y, new_pos, t);
        break;
      case T_DIRECTION:
        new_pos = (front_back == FRONT) ? ((t+1)%grid_t) : ((t+grid_t-1)%grid_t);
        return Coord(x, y, z, new_pos);
        break;

      default:
        break;
    }
    return *this;
  }

  Coord& operator=(const Coord& rhs) {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    t = rhs.t;
    return *this;
  }
};
static Coord coord;




class Point {
private:
  int x_;
  int y_;
  int z_;
  int t_;
  int parity_;
public:
  Point() = default;
  __device__ Point(const Point& rhs) : x_(rhs.x_), y_(rhs.y_), z_(rhs.z_), t_(rhs.t_), parity_(rhs.parity_) {}
  __device__ Point(int x, int y, int z, int t, int parity) : x_(x), y_(y), z_(z), t_(t), parity_(parity) {}
  __device__ void outputInfo() {
    printf("Point: (x,y,z,t)=(%d, %d, %d, %d), parity = %d\n", x_, y_, z_, t_, parity_);
  }
  __device__ int getParity() const { return parity_;}
  __device__ Point move(int front_back, int direction, int Lx, int Ly, int Lz, int Lt) const{ // direction +-1234
    // 1-front 0-back
    assert(abs(direction) >= 0 && abs(direction) < 4);
    assert(front_back == BACK || front_back == FRONT);

    int new_pos;
    int eo = (y_ + z_ + t_) & 0x01;    // (y+z+t)%2

    if (direction == 0) {
      if (!front_back) {
        new_pos = x_ + (eo == parity_) * (-1 + (x_ == 0) * Lx);
        return Point(new_pos, y_, z_, t_, 1-parity_);
      } else {
        new_pos = x_ + (eo != parity_) * (1 + (x_ == Lx-1) * (-Lx));
        return Point(new_pos, y_, z_, t_, 1-parity_);
      }
    } else if (direction == 1) {  // y 前进
      if (!front_back) {
        new_pos = y_ - 1 + (y_ == 0) * Ly;
        return Point(x_, new_pos, z_, t_, 1-parity_);
      } else {
        new_pos = y_ + 1 + (y_ == Ly-1) * (-Ly);
        return Point(x_, new_pos, z_, t_, 1-parity_);
      }
    } else if (direction == 2) {
      if (!front_back) {
        new_pos = z_ - 1 + (z_ == 0) * Lz;
        return Point(x_, y_, new_pos, t_, 1-parity_);
      } else {
        new_pos = z_ + 1 + (z_ == Lz-1) * (-Lz);
        return Point(x_, y_, new_pos, t_, 1-parity_);
      }
    } else if (direction == 3) {
      if (!front_back) {
        new_pos = t_ - 1 + (t_ == 0) * Lt;
        return Point(x_, y_, z_, new_pos, 1-parity_);
      } else {
        new_pos = t_ + 1 + (t_ == Lt-1) * (-Lt);
        return Point(x_, y_, z_, new_pos, 1-parity_);
      }
    } else {
      return *this;
    }
  }

  __device__ Complex* getPointGauge(Complex* origin, int direction, int Lx, int Ly, int Lz, int Lt) const{
    return origin + (((((((direction << 1) + parity_) * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * Nc * Nc;
  }

  __device__ Complex* getPointVector(Complex* origin, int Lx, int Ly, int Lz, int Lt) const{
    return origin + (((t_ * Lz + z_) * Ly + y_) * Lx + x_) * Ns * Nc;
  }
  __device__ Point& operator= (const Point& rhs) {
    x_ = rhs.x_;
    y_ = rhs.y_;
    z_ = rhs.z_;
    t_ = rhs.t_;
    parity_ = rhs.parity_;
    return *this;
  }
  __device__ Complex* getPointClover(Complex* origin, int Lx, int Ly, int Lz, int Lt) const{
    return origin + (((((parity_ * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * (Nc * Ns * Nc * Ns / 2);
  }
};

// FROM newbing
// 定义一个函数，用于交换两行
__device__ void swapRows(Complex* matrix, int row1, int row2) {
  Complex temp;
  for (int i = 0; i < N * 2; i++) {
    temp = matrix[row1 * N * 2 + i];
    matrix[row1 * N * 2 + i] = matrix[row2 * N * 2 + i];
    matrix[row2 * N * 2 + i] = temp;
  }
}
// 定义一个函数，用于将一行除以一个数
__device__ void divideRow(Complex* matrix, int row, Complex num) {
  for (int i = 0; i < N * 2; i++) {
    matrix[row * N * 2 + i] /= num;
  }
}
// 定义一个函数，用于将一行减去另一行乘以一个数
__device__ void subtractRow(Complex* matrix, int row1, int row2, Complex num) {
  for (int i = 0; i < N * 2; i++) {
    matrix[row1 * N * 2 + i] -= num * matrix[row2 * N * 2 + i];
  }
}
// 定义一个函数，用于打印矩阵
__device__ void printMatrix(Complex* matrix) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N * 2; j++) {
      matrix[i * N * 2 + j].output();
    }
    printf("\n");
  }
}
// 定义一个函数，用于求逆矩阵
__device__ void inverseMatrix(Complex* matrix, Complex* result) {
  // 创建一个单位矩阵
  // double* identity = (double*)malloc(sizeof(double) * N * N);
  Complex identity[N*N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        identity[i * N + j] = Complex(1, 0);
      } else {
        identity[i * N + j].clear2Zero();
      }
    }
  }
  // 将原矩阵和单位矩阵平接在一起，形成增广矩阵
  Complex augmented[N*N*2];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N * 2; j++) {
      if (j < N) {
        augmented[i * N * 2 + j] = matrix[i * N + j];
      } else {
        augmented[i * N * 2 + j] = identity[i * N + (j - N)];
      }
    }
  }

  // 对增广矩阵进行高斯消元法
  for (int i = 0; i < N; i++) {
    // 如果对角线上的元素为0，就交换两行
    if (augmented[i * N * 2 + i] == Complex(0, 0)) {
      for (int j = i + 1; j < N; j++) {
        if (augmented[j * N * 2 + i] != Complex(0,0)) {
          swapRows(augmented, i, j);
          break;
        }
      }
    }

    // 如果对角线上的元素不为1，就将该行除以该元素
    if (augmented[i * N * 2 + i] != Complex(1,0)) {
      divideRow(augmented, i, augmented[i * N * 2 + i]);
    }

    // 将其他行减去该行乘以相应的系数，使得该列除了对角线上的元素外都为0
    for (int j = 0; j < N; j++) {
      if (j != i) {
        subtractRow(augmented, j, i, augmented[j * N * 2 + i]);
      }
    }
  }

  // 从增广矩阵中分离出逆矩阵
  // double* inverse = (double*)malloc(sizeof(double) * N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // inverse[i * N + j] = augmented[i * N * 2 + (j + N)];
      result[i * N + j] = augmented[i * N * 2 + (j + N)];
    }
  }
}

// end (FROM  newbing)

__device__ inline void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}
__device__ inline void loadGauge(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* u = p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
__device__ inline void loadVector(Complex* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}

// __device__ void storeVectorBySharedMemory(Complex* origin, Complex* result) {
//   // result is register variable
//   __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
//   Complex* block_dst = origin + blockDim.x * blockIdx.x * Ns * Nc;
//   double* double_dst = reinterpret_cast<double*>(block_dst);
//   Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;

//   // store result to shared memory
//   for (int i = 0; i < Ns * Nc; i++) {
//     shared_dst[i] = result[i];
//   }
//   __syncthreads();
//   // store result of shared memory to global memory
//   for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
//     double_dst[i] = shared_buffer[i];
//   }
// }

// WARP version, no sync
__device__ void storeVectorBySharedMemory(void* shared_ptr, Complex* origin, Complex* result) {
  // result is register variable
  // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
    // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
  double* shared_buffer = static_cast<double*>(shared_ptr);
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;
  Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;
  Complex* warp_dst = origin + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc;
  double* double_dst = reinterpret_cast<double*>(warp_dst);

  // store result to shared memory
  for (int i = 0; i < Ns * Nc; i++) {
    shared_dst[i] = result[i];
  }
  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    double_dst[i] = shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i];
  }
}

// __device__ void loadVectorBySharedMemory(Complex* origin, Complex* src_local) {
//   // src_local is register variable
//   __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
//   Complex* block_src = origin + blockDim.x * blockIdx.x * Ns * Nc;
//   double* double_src = reinterpret_cast<double*>(block_src);
//   Complex* shared_src = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;

//   // load from global memory
//   for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
//     shared_buffer[i] = double_src[i];
//   }
//   __syncthreads();
//   // load src from shared memory
//   for (int i = 0; i < Ns * Nc; i++) {
//     src_local[i] = shared_src[i];
//   }
// }

// WARP version, no sync
__device__ void loadVectorBySharedMemory(void* shared_ptr, Complex* origin, Complex* result) {
  // result is register variable
  // __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
  double* shared_buffer = static_cast<double*>(shared_ptr);
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_index = (thread - thread / BLOCK_SIZE * BLOCK_SIZE) / WARP_SIZE;//thread % BLOCK_SIZE / WARP_SIZE;
  Complex* shared_dst = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;
  Complex* warp_dst = origin + (thread / WARP_SIZE * WARP_SIZE) * Ns * Nc;
  double* double_dst = reinterpret_cast<double*>(warp_dst);

  // store result of shared memory to global memory
  for (int i = threadIdx.x - threadIdx.x / WARP_SIZE * WARP_SIZE; i < WARP_SIZE * Ns * Nc * 2; i += WARP_SIZE) {
    shared_buffer[warp_index * WARP_SIZE * Ns * Nc * 2 + i] = double_dst[i];
  }
  // store result to shared memory
  for (int i = 0; i < Ns * Nc; i++) {
    result[i] = shared_dst[i];
  }
}

// __device__ void loadCloverBySharedMemory(Complex* origin, Complex* src_local) {
//   // src_local is register variable
//   __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
//   Complex* block_src = origin + blockDim.x * blockIdx.x * Ns * Nc;
//   double* double_src = reinterpret_cast<double*>(block_src);
//   Complex* shared_src = reinterpret_cast<Complex*>(shared_buffer) + threadIdx.x * Ns * Nc;

//   // load from global memory
//   for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
//     shared_buffer[i] = double_src[i];
//   }
//   __syncthreads();
//   // load src from shared memory
//   for (int i = 0; i < Ns * Nc; i++) {
//     src_local[i] = shared_src[i];
//   }
// }



__global__ void DslashTransferFrontX(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z+t) & 0x01;
  Point p(sub_Lx-1, 2 * sub_y + (new_even_odd == 1-parity), z, t, 1-parity);

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) *
            u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
            u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1);
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void DslashTransferBackX(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;   // 注意这里乘以2
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;

  int new_even_odd = (z+t) & 0x01;
  Point p(0, 2 * sub_y + (new_even_odd != 1-parity), z, t, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}
__global__ void DslashTransferFrontY(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly-1, z, t, 1-parity);

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp;
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}
__global__ void DslashTransferBackY(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}
// DslashTransferFrontZ: DONE
__global__ void DslashTransferFrontZ(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz-1, t, 1-parity);

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1);
    }
  }


  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

}
// DslashTransferBackZ: Done
__global__ void DslashTransferBackZ(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, 0, t, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}

// DslashTransferFrontT: Done
__global__ void DslashTransferFrontT(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer){
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt-1, 1-parity);

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  dst_ptr = send_buffer + thread*Ns*Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp;
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }


}
// DslashTransferBackT: Done
__global__ void DslashTransferBackT(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, 0, 1-parity);

  Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  dst_ptr = send_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = src_local[i];
  }
}

__global__ void calculateBackBoundaryX(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;
  
  int new_even_odd = (z+t) & 0x01;  // %2
  Point p(0, 2 * sub_y + (new_even_odd != parity), z, t, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

}
__global__ void calculateFrontBoundaryX(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;   // 注意这里乘以2
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;
  
  int new_even_odd = (z+t) & 0x01;  // %2
  Point p(sub_Lx-1, 2 * sub_y + (new_even_odd == parity), z, t, parity);

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1);
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateBackBoundaryY(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateFrontBoundaryY(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly-1, z, t, parity);

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp;
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}
__global__ void calculateBackBoundaryZ(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, 0, t, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateFrontBoundaryZ(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz-1, t, parity);

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1);
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

}

__global__ void calculateBackBoundaryT(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, 0, parity);

  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  src_ptr = recv_buffer + thread*Ns*Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}

__global__ void calculateFrontBoundaryT(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt-1, parity);

  Complex temp;
  Complex* src_ptr;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, 3, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = recv_buffer + thread * Ns * Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp;
    }
  }

  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}


__global__ void mpiDslash(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t) {
  assert(parity == 0 || parity == 1);

  __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];
  Lx >>= 1;

  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int coord_boundary;
  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex temp;
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, 1);
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, 1);
      }
    }
  }

  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 0, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, -1);
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, -1);
      }
    }
  }

  // \mu = 2
  // y front
  loadGauge(u_local, gauge, 1, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += -temp;
      }
    }
  }

  // z back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 1, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[3 * Nc + i] += -temp;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp;
      }
    }
  }

  // \mu = 3
  // z front
  loadGauge(u_local, gauge, 2, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, 1);
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, -1);
      }
    }
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 2, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp * Complex(0, -1);
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) *
              u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp * Complex(0, 1);
      }
    }
  }

  // t: front
  loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += -temp;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += -temp;
      }
    }
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 3, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[0 * Nc + i] += temp;
        dst_local[2 * Nc + i] += temp;
        // second row vector with col vector
        temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
        dst_local[1 * Nc + i] += temp;
        dst_local[3 * Nc + i] += temp;
      }
    }
  }

  // store result
  storeVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), dst_local);
  // double *dest = static_cast<double *>(fermion_out) + (blockIdx.x * BLOCK_SIZE) * Ns * Nc * 2;
  // double *dest_temp_double = (double *)dst_local;
  // for (int i = 0; i < Ns * Nc * 2; i++) {
  //   shared_output_vec[threadIdx.x * Ns * Nc * 2 + i] = dest_temp_double[i];
  // }
  // __syncthreads();
  // // load to global memory
  // for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
  //   dest[i] = shared_output_vec[i];
  // }
}
__global__ void gpuDslash(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity)
{
  assert(parity == 0 || parity == 1);

  __shared__ double shared_output_vec[BLOCK_SIZE * Ns * Nc * 2];
  Lx >>= 1;

  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU

  Point p(x, y, z, t, parity);
  Point move_point;


  Complex temp;
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGauge(u_local, gauge, 0, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1);
    }
  }

  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 0, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1);
    }
  }

  // \mu = 2
  loadGauge(u_local, gauge, 1, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j]) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp;
    }
  }

  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 1, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp;
    }
  }

  // \mu = 3
  loadGauge(u_local, gauge, 2, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1);
    }
  }

  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 2, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1);
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1);
    }
  }

  // \mu = 4
  loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j]) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j]) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp;
    }
  }

  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, 3, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp;
    }
  }

  // store result
  double *dest = static_cast<double *>(fermion_out) + (blockIdx.x * BLOCK_SIZE) * Ns * Nc * 2;
  double *dest_temp_double = (double *)dst_local;
  for (int i = 0; i < Ns * Nc * 2; i++) {
    shared_output_vec[threadIdx.x * Ns * Nc * 2 + i] = dest_temp_double[i];
  }
  __syncthreads();
  // load to global memory
  for (int i = threadIdx.x; i < BLOCK_SIZE * Ns * Nc * 2; i += BLOCK_SIZE) {
    dest[i] = shared_output_vec[i];
  }
}

__device__ void copyGauge (Complex* dst, Complex* src) {
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    dst[i] = src[i];
  }
  reconstructSU3(dst);
}



__global__ void sendGaugeBoundaryToBufferX(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly);
  int z = thread % (Lz * Ly) / Ly;
  int y = thread % Ly;

  int x;
  int sub_x;
  int single_gauge = Ly * Lz * Lt * Nc * Nc;
  Complex* src;
  Complex* dst;

  // forward
  x = Lx - 1;
  sub_x = x >> 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
    copyGauge(dst, src);
  }

  // backward
  x = 0;
  sub_x = x >> 1;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
    copyGauge(dst, src);
  }
}

__global__ void sendGaugeBoundaryToBufferY(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Lx);
  int z = thread % (Lz * Lx) / Lx;
  int x = thread % Lx;
  int sub_x = x >> 1;
  int single_gauge = Lx * Lz * Lt * Nc * Nc;
  int y;
  Complex* src;
  Complex* dst;

  // forward
  y = Ly - 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);

  }

  // backward
  y = 0;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);
  }
}
__global__ void sendGaugeBoundaryToBufferZ(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;
  int sub_x = x >> 1;
  int single_gauge = Lx * Ly * Lt * Nc * Nc;
  int z;
  Complex* src;
  Complex* dst;

  // forward
  z = Lz - 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);

  }

  // backward
  z = 0;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);
  }
}
__global__ void sendGaugeBoundaryToBufferT(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt) {
  int parity;
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;
  int sub_x = x >> 1;
  int single_gauge = Lx * Ly * Lz * Nc * Nc;
  int t;
  Complex* src;
  Complex* dst;

  // forward
  t = Lt - 1;
  parity = (x + y + z + t) & 0x01;
  Point f_p(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(front_buffer) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);

  }

  // backward
  t = 0;
  parity = (x + y + z + t) & 0x01;
  f_p = Point(sub_x, y, z, t, parity);
  for (int i = 0; i < Nd; i++) {
    src = f_p.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
    dst = static_cast<Complex*>(back_buffer) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
    copyGauge(dst, src);
  }
}

__global__ void shiftGaugeX(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Ly * Lz * Lt * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  if (x < Lx - 1) {
    move_point = Point((x+1) >> 1, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    move_point = Point(0, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  if (x > 0) {
    move_point = Point((x-1) >> 1, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point((Lx-1)>>1, y, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((t * Lz + z) * Ly + y) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}

__global__ void shiftGaugeY(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Lx * Lz * Lt * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  if (y < Ly - 1) {
    move_point = Point(sub_x, y+1, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // forward boundary to Lt-1 line
    move_point = Point(sub_x, 0, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  // if (t < Lt-1) {
  if (y > 0) {
    move_point = Point(sub_x, y-1, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point(sub_x, Ly-1, z, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((t * Lz + z) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}

__global__ void shiftGaugeZ(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Lx * Ly * Lt * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  if (z < Lz - 1) {
    move_point = Point(sub_x, y, z+1, t, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // forward boundary to Lt-1 line
    // move_point = Point(sub_x, y, z, Lt-1, 1-parity);
    move_point = Point(sub_x, y, 0, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  if (z > 0) {
    move_point = Point(sub_x, y, z-1, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point(sub_x, y, Lz-1, t, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((t * Ly + y) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}


__global__ void shiftGaugeT(void* origin_gauge, void* front_shift_gauge, void* back_shift_gauge, void* front_boundary, void* back_boundary, int Lx, int Ly, int Lz, int Lt) {
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  int sub_x = (x >> 1);
  int single_gauge = Lx * Ly * Lz * Nc * Nc;
  int parity = (x + y + z + t) & 0x01;

  Complex* src;
  Complex* dst;

  Point point(sub_x, y, z, t, parity);
  Point move_point;

  // backward-shift
  // if (t > 0) {
  if (t < Lt - 1) {
    // move_point = Point(sub_x, y, z, t-1, 1-parity);
    move_point = Point(sub_x, y, z, t+1, 1-parity);
    for (int i = 0; i < Nd; i++) {  // i means four dims
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // forward boundary to Lt-1 line
    // move_point = Point(sub_x, y, z, Lt-1, 1-parity);
    move_point = Point(sub_x, y, z, 0, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(back_boundary) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
      // dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(back_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }

  // forward-shift
  // if (t < Lt-1) {
  if (t > 0) {
    move_point = Point(sub_x, y, z, t-1, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = point.getPointGauge(static_cast<Complex*>(origin_gauge), i, sub_Lx, Ly, Lz, Lt);
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  } else {
    // backward boundary to 0 line
    move_point = Point(sub_x, y, z, Lt-1, 1-parity);
    for (int i = 0; i < Nd; i++) {
      src = static_cast<Complex*>(front_boundary) + i * single_gauge + ((z * Ly + y) * Lx + x) * Nc * Nc;
      dst = move_point.getPointGauge(static_cast<Complex*>(front_shift_gauge), i, sub_Lx, Ly, Lz, Lt);
      copyGauge(dst, src);
    }
  }
}

class MPICommunicator;
static MPICommunicator *mpi_comm;

class MPICommunicator {
private:
  int Lx_;
  int Ly_;
  int Lz_;
  int Lt_;
  int grid_front[Nd];
  int grid_back[Nd];

  Complex* gauge_;
  Complex* fermion_in_;
  Complex* fermion_out_;

  Complex* d_send_front_vec[Nd];   // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively
  Complex* d_send_back_vec[Nd];    // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively
  Complex* d_recv_front_vec[Nd];   // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively
  Complex* d_recv_back_vec[Nd];    // device pointer: 0, 1, 2, 3 represent x, y, z, t aspectively

  Complex* h_send_front_vec[Nd];   // host pointer
  Complex* h_send_back_vec[Nd];    // host pointer
  Complex* h_recv_front_vec[Nd];   // host pointer
  Complex* h_recv_back_vec[Nd];    // host pointer

  Complex* gauge_shift[Nd][2];    // Nd - 4 dims    2: Front/back
  Complex* gauge_twice_shift[6][4];    // 6-combination   4: ++ +- -+ --

  Complex* h_send_gauge[Nd][2];   // Nd - 4 dims    2: Front/back
  Complex* d_send_gauge[Nd][2];   // Nd - 4 dims    2: Front/back
  Complex* h_recv_gauge[Nd][2];   // Nd - 4 dims    2: Front/back
  Complex* d_recv_gauge[Nd][2];   // Nd - 4 dims    2: Front/back

  MPI_Request send_front_req[Nd];
  MPI_Request send_back_req[Nd];
  MPI_Request recv_front_req[Nd];
  MPI_Request recv_back_req[Nd];

  MPI_Status send_front_status[Nd];
  MPI_Status send_back_status[Nd];
  MPI_Status recv_front_status[Nd];
  MPI_Status recv_back_status[Nd];
public:
  Complex* getOriginGauge() {
    return gauge_;
  }
  Complex** getShiftGauge() {
    return &(gauge_shift[0][0]);
  }
  Complex** getShiftShiftGauge() {
    return &(gauge_twice_shift[0][0]);
  }

  void prepareGauge() {

    for (int i = 0; i < Nd; i++) {
      shiftGauge(gauge_, gauge_shift[i][FRONT], gauge_shift[i][BACK], i);
    }

    // twice shift
    // xy-0    xz-1    xt-2
    // yz-3    yt-4    zt-5
    // 0 means ++     1 means +-     2 means -+     3 means --
    shiftGauge(gauge_shift[X_DIRECTION][FRONT], gauge_twice_shift[0][0], gauge_twice_shift[0][1], Y_DIRECTION);
    shiftGauge(gauge_shift[X_DIRECTION][BACK], gauge_twice_shift[0][2], gauge_twice_shift[0][3], Y_DIRECTION);

    shiftGauge(gauge_shift[X_DIRECTION][FRONT], gauge_twice_shift[1][0], gauge_twice_shift[1][1], Z_DIRECTION);
    shiftGauge(gauge_shift[X_DIRECTION][BACK], gauge_twice_shift[1][2], gauge_twice_shift[1][3], Z_DIRECTION);

    shiftGauge(gauge_shift[X_DIRECTION][FRONT], gauge_twice_shift[2][0], gauge_twice_shift[2][1], T_DIRECTION);
    shiftGauge(gauge_shift[X_DIRECTION][BACK], gauge_twice_shift[2][2], gauge_twice_shift[2][3], T_DIRECTION);

    shiftGauge(gauge_shift[Y_DIRECTION][FRONT], gauge_twice_shift[3][0], gauge_twice_shift[3][1], Z_DIRECTION);
    shiftGauge(gauge_shift[Y_DIRECTION][BACK], gauge_twice_shift[3][2], gauge_twice_shift[3][3], Z_DIRECTION);

    shiftGauge(gauge_shift[Y_DIRECTION][FRONT], gauge_twice_shift[4][0], gauge_twice_shift[4][1], T_DIRECTION);
    shiftGauge(gauge_shift[Y_DIRECTION][BACK], gauge_twice_shift[4][2], gauge_twice_shift[4][3], T_DIRECTION);

    shiftGauge(gauge_shift[Z_DIRECTION][FRONT], gauge_twice_shift[5][0], gauge_twice_shift[5][1], T_DIRECTION);
    shiftGauge(gauge_shift[Z_DIRECTION][BACK], gauge_twice_shift[5][2], gauge_twice_shift[5][3], T_DIRECTION);
    // printf(RED"shift gauge over");

    printf(CLR"");
  }
  void shiftGauge(void* src_gauge, void* front_shift_gauge, void* back_shift_gauge, int direction) {
    // printf("");
    prepareBoundaryGauge(src_gauge, direction);      // 2nd and 3rd parameter ----send buffer
    sendGauge(direction);
    recvGauge(direction);
    qcuGaugeMPIBarrier(direction);
    shiftGaugeKernel(src_gauge, front_shift_gauge, back_shift_gauge, direction);
  }

  void shiftGaugeKernel(void* src_gauge, void* front_shift_gauge, void* back_shift_gauge, int direction) {
    int vol = Lx_ * Ly_ * Lz_ * Lt_;
    dim3 gridDim(vol / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    void *args[] = {&src_gauge, &front_shift_gauge, &back_shift_gauge, &d_recv_gauge[direction][FRONT], &d_recv_gauge[direction][BACK], &Lx_, &Ly_, &Lz_, &Lt_};
    switch(direction) {
      case X_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeX, gridDim, blockDim, args));
        break;
      case Y_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeY, gridDim, blockDim, args));
        break;
      case Z_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeZ, gridDim, blockDim, args));
        break;
      case T_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)shiftGaugeT, gridDim, blockDim, args));
        break;
      default:
        break;
    }
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void prepareBoundaryGauge(void* src_gauge, int direction) {
    void *args[] = {&src_gauge, &d_send_gauge[direction][FRONT], &d_send_gauge[direction][BACK], &Lx_, &Ly_, &Lz_, &Lt_};
    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[direction] = 1;
    int full_length = 1;
    for (int i = 0; i < Nd; i++) {
      full_length *= length[i];
    }

    dim3 gridDim(full_length / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    // sendGaugeBoundaryToBufferX(void* origin_gauge, void* front_buffer, void* back_buffer, int Lx, int Ly, int Lz, int Lt)
    switch (direction) {
      case X_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferX, gridDim, blockDim, args));
        break;
      case Y_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferY, gridDim, blockDim, args));
        break;
      case Z_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferZ, gridDim, blockDim, args));
        break;
      case T_DIRECTION:
        checkCudaErrors(cudaLaunchKernel((void *)sendGaugeBoundaryToBufferT, gridDim, blockDim, args));
        break;
      default:
        break;
    }
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void qcuGaugeMPIBarrier(int direction) {
    int process;
    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[direction] = 1;
    int boundary_length = 1;
    for (int i = 0; i < Nd; i++) {
      boundary_length *= length[i];
    }
    boundary_length *= (Nd * Nc * Nc);
    // from front process
    process = grid_front[direction];
    if (process_rank != process) {
      // recv
      MPI_Wait(&recv_front_req[direction], &recv_front_status[direction]);
      // send
      MPI_Wait(&send_front_req[direction], &send_front_status[direction]);
      checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][FRONT], h_recv_gauge[direction][FRONT], sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
    } else {
      checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][FRONT], d_send_gauge[direction][BACK], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToDevice));
    }
    // from back process
    process = grid_back[direction];
    if (process_rank != process) {
      // recv
      MPI_Wait(&recv_back_req[direction], &recv_back_status[direction]);
      // send
      MPI_Wait(&send_back_req[direction], &send_back_status[direction]);
      checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][BACK], h_recv_gauge[direction][BACK], sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
    } else {
      checkCudaErrors(cudaMemcpy(d_recv_gauge[direction][BACK], d_send_gauge[direction][FRONT], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToDevice));
    }
  }
  void recvGauge(int direction) {
    int src_process;
    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[direction] = 1;
    int boundary_length = 1;
    for (int i = 0; i < Nd; i++) {
      boundary_length *= length[i];
    }

    boundary_length *= (Nd * Nc * Nc);
    src_process = grid_front[direction];
    if (process_rank != src_process) {  // send buffer to other process
      MPI_Irecv(h_recv_gauge[direction][FRONT], boundary_length*2, MPI_DOUBLE, src_process, BACK, MPI_COMM_WORLD, &recv_front_req[direction]);
    }
    // back
    src_process = grid_back[direction];
    if (process_rank != src_process) {  // send buffer to other process
      MPI_Irecv(h_recv_gauge[direction][BACK], boundary_length*2, MPI_DOUBLE, src_process, FRONT, MPI_COMM_WORLD, &recv_back_req[direction]);
    }
  }
  void sendGauge(int direction) {
    int dst_process;
    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[direction] = 1;
    int boundary_length = 1;
    for (int i = 0; i < Nd; i++) {
      boundary_length *= length[i];
    }
    boundary_length *= (Nd * Nc * Nc);
    // front
    dst_process = grid_front[direction];
    if (process_rank != dst_process) {  // send buffer to other process
      checkCudaErrors(cudaMemcpy(h_send_gauge[direction][FRONT], d_send_gauge[direction][FRONT], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_send_gauge[direction][FRONT], boundary_length*2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
      }
    // back
    dst_process = grid_back[direction];
    if (process_rank != dst_process) {  // send buffer to other process
      checkCudaErrors(cudaMemcpy(h_send_gauge[direction][BACK], d_send_gauge[direction][BACK], sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_send_gauge[direction][BACK], boundary_length*2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
    }
  }

  void allocateGaugeBuffer() {
    int total_vol = Nd * Lt_ * Lz_ * Ly_ * Lx_ * Nc * Nc;
    // TODO: allocate  gauge 
    for (int i = 0; i < Nd; i++) {
      for (int j = 0; j < 2; j++) {
        checkCudaErrors(cudaMalloc(&gauge_shift[i][j], sizeof(Complex) * total_vol));
      }
    }
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 4; j++) {
        checkCudaErrors(cudaMalloc(&gauge_twice_shift[i][j], sizeof(Complex) * total_vol));
      }
    }

    // TODO: allocate boundary gauge
    int boundary_size;
    // int length[4] = {Lx_, Ly_, Lz_, Lt_};
    for (int i = 0; i < Nd; i++) {
      // int shift[Nd] = {1, 1, 1, 1};
      int length[4] = {Lx_, Ly_, Lz_, Lt_};
      length[i] = 1;
      boundary_size = 1;
      for (int j = 0; j < Nd; j++) {
        boundary_size *= length[j];
      }
      boundary_size *= (Nd * Nc * Nc);
      for (int j = 0; j < 2; j++) {
        h_send_gauge[i][j] = new Complex[boundary_size];
        h_recv_gauge[i][j] = new Complex[boundary_size];
        checkCudaErrors(cudaMalloc(&d_send_gauge[i][j], sizeof(Complex) * boundary_size));
        checkCudaErrors(cudaMalloc(&d_recv_gauge[i][j], sizeof(Complex) * boundary_size));
      }
    }
  }

  MPICommunicator (Complex* gauge, Complex* fermion_in, Complex* fermion_out, int Lx, int Ly, int Lz, int Lt) : gauge_(gauge), fermion_in_(fermion_in), fermion_out_(fermion_out), Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt) {
    for (int i = 0; i < Nd; i++) {
      d_send_front_vec[i] = nullptr;
      d_send_back_vec[i] = nullptr;
      d_recv_front_vec[i] = nullptr;
      d_recv_back_vec[i] = nullptr;

      h_send_front_vec[i] = nullptr;
      h_send_back_vec[i] = nullptr;
      h_recv_front_vec[i] = nullptr;
      h_recv_back_vec[i] = nullptr;
    }
    allocateBuffer();
    calculateAdjacentProcess();
    allocateGaugeBuffer();
    prepareGauge();
  }
  ~MPICommunicator() {
    for (int i = 0; i < Nd; i++) {
      // free host pointer
      if (h_send_front_vec[i]) {
        delete h_send_front_vec[i];
        h_send_front_vec[i] = nullptr;
      }
      if (h_send_back_vec[i]) {
        delete h_send_back_vec[i];
        h_send_back_vec[i] = nullptr;
      }
      if (h_recv_front_vec[i]) {
        delete h_recv_front_vec[i];
        h_recv_front_vec[i] = nullptr;
      }
      if (h_recv_back_vec[i]) {
        delete h_recv_back_vec[i];
        h_recv_back_vec[i] = nullptr;
      }
      // free device pointer
      // if (d_send_front_vec[i] != nullptr) {
      //   checkCudaErrors(cudaFree(d_send_front_vec[i]));
      //   d_send_front_vec[i] = nullptr;
      // }
      // if (d_send_back_vec[i] != nullptr) {
      //   checkCudaErrors(cudaFree(d_send_back_vec[i]));
      //   d_send_back_vec[i] = nullptr;
      // }
      // if (d_recv_front_vec[i] != nullptr) {
      //   checkCudaErrors(cudaFree(d_recv_front_vec[i]));
      //   d_recv_front_vec[i] = nullptr;
      // }
      // if (d_recv_back_vec[i] != nullptr) {
      //   checkCudaErrors(cudaFree(d_recv_back_vec[i]));
      //   d_recv_back_vec[i] = nullptr;
      // }
    }
  }
  MPICommunicator (void* gauge, void* fermion_in, void* fermion_out, int Lx, int Ly, int Lz, int Lt) : gauge_(static_cast<Complex*>(gauge)), fermion_in_(static_cast<Complex*>(fermion_in)), fermion_out_(static_cast<Complex*>(fermion_out)), Lx_(Lx), Ly_(Ly), Lz_(Lz), Lt_(Lt){
    for (int i = 0; i < Nd; i++) {
      d_send_front_vec[i] = nullptr;
      d_send_back_vec[i] = nullptr;
      d_recv_front_vec[i] = nullptr;
      d_recv_back_vec[i] = nullptr;
      h_send_front_vec[i] = nullptr;
      h_send_back_vec[i] = nullptr;
      h_recv_front_vec[i] = nullptr;
      h_recv_back_vec[i] = nullptr;
    }
    allocateBuffer();
    calculateAdjacentProcess();
    allocateGaugeBuffer();
    prepareGauge();
  }

  void allocateBuffer() {
    int boundary_size;
    if (grid_x != 1) {
      boundary_size = Ly_ * Lz_ * Lt_ * Ns * Nc / 2;
      h_send_front_vec[0] = new Complex[boundary_size];
      h_send_back_vec[0] = new Complex[boundary_size];
      h_recv_front_vec[0] = new Complex[boundary_size];
      h_recv_back_vec[0] = new Complex[boundary_size];
      checkCudaErrors(cudaMalloc(&d_send_front_vec[0], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_send_back_vec[0], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_front_vec[0], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_back_vec[0], sizeof(Complex) * boundary_size));
    }
    if (grid_y != 1) {
      boundary_size = Lx_ * Lz_ * Lt_ * Ns * Nc / 2;
      h_send_front_vec[1] = new Complex[boundary_size];
      h_send_back_vec[1] = new Complex[boundary_size];
      h_recv_front_vec[1] = new Complex[boundary_size];
      h_recv_back_vec[1] = new Complex[boundary_size];
      checkCudaErrors(cudaMalloc(&d_send_front_vec[1], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_send_back_vec[1], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_front_vec[1], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_back_vec[1], sizeof(Complex) * boundary_size));
    }
    if (grid_z != 1) {
      boundary_size = Lx_ * Ly_ * Lt_ * Ns * Nc / 2;
      h_send_front_vec[2] = new Complex[boundary_size];
      h_send_back_vec[2] = new Complex[boundary_size];
      h_recv_front_vec[2] = new Complex[boundary_size];
      h_recv_back_vec[2] = new Complex[boundary_size];
      checkCudaErrors(cudaMalloc(&d_send_front_vec[2], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_send_back_vec[2], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_front_vec[2], sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&d_recv_back_vec[2], sizeof(Complex) * boundary_size));
    }
    if (grid_t != 1) {
      boundary_size = Lx_ * Ly_ * Lz_ * Ns * Nc / 2;
      h_send_front_vec[3] = new Complex[boundary_size];
      h_send_back_vec[3] = new Complex[boundary_size];
      h_recv_front_vec[3] = new Complex[boundary_size];
      h_recv_back_vec[3] = new Complex[boundary_size];
      checkCudaErrors(cudaMalloc(&(d_send_front_vec[3]), sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&(d_send_back_vec[3]), sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&(d_recv_front_vec[3]), sizeof(Complex) * boundary_size));
      checkCudaErrors(cudaMalloc(&(d_recv_back_vec[3]), sizeof(Complex) * boundary_size));
    }
  }
  void calculateAdjacentProcess() {
    grid_front[0] = coord.adjCoord(FRONT, X_DIRECTION).calculateMpiRank();
    grid_back[0] = coord.adjCoord(BACK, X_DIRECTION).calculateMpiRank();
    grid_front[1] = coord.adjCoord(FRONT, Y_DIRECTION).calculateMpiRank();
    grid_back[1] = coord.adjCoord(BACK, Y_DIRECTION).calculateMpiRank();
    grid_front[2] = coord.adjCoord(FRONT, Z_DIRECTION).calculateMpiRank();
    grid_back[2] = coord.adjCoord(BACK, Z_DIRECTION).calculateMpiRank();
    grid_front[3] = coord.adjCoord(FRONT, T_DIRECTION).calculateMpiRank();
    grid_back[3] = coord.adjCoord(BACK, T_DIRECTION).calculateMpiRank();
  }
  int getAdjacentProcess (int front_back, int direction) {
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back == FRONT) {
      return grid_front[direction];
    } else {
      return grid_back[direction];
    }
  }
  // return device pointer
  Complex* getSendBufferAddr(int front_back, int direction) { // TO SEND
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return d_send_front_vec[direction];
    }
    else {
      return d_send_back_vec[direction];
    }
  }
  Complex* getHostSendBufferAddr(int front_back, int direction) { // TO SEND
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return h_send_front_vec[direction];
    }
    else {
      return h_send_back_vec[direction];
    }
  }
  Complex* getRecvBufferAddr(int front_back, int direction) { // TO RECEIVE
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return d_recv_front_vec[direction];
    }
    else {
      return d_recv_back_vec[direction];
    }
  }
  Complex* getHostRecvBufferAddr(int front_back, int direction) { // TO RECEIVE
    assert(front_back==FRONT || front_back==BACK);
    assert(direction==X_DIRECTION || direction==Y_DIRECTION || direction==Z_DIRECTION || direction==T_DIRECTION);
    if (front_back==FRONT) {
      return h_recv_front_vec[direction];
    }
    else {
      return h_recv_back_vec[direction];
    }
  }

  void preDslash(void* fermion_in, int parity) {
    for (int i = 0; i < Nd; i++) {
      // calc Boundary and send Boundary
      prepareFrontBoundaryVector (fermion_in, i, parity);
    }
    for (int i = 0; i < Nd; i++) {
      // recv Boundary
      recvBoundaryVector(i);
    }
  }
  void postDslash(void* fermion_out, int parity) {
    Complex* h_addr;
    Complex* d_addr;
    int boundary_length;

    // Barrier
    for (int i = 0; i < Nd; i++) {
      int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
      length[i] = 1;
      int sub_vol = 1;
      for (int j = 0; j < Nd; j++) {
        sub_vol *= length[j];
      }
      sub_vol >>= 1;  // div 2
      boundary_length = sub_vol * (Ns * Nc);

      dim3 subGridDim(sub_vol / BLOCK_SIZE);
      dim3 blockDim(BLOCK_SIZE);

      h_addr = mpi_comm->getHostRecvBufferAddr(FRONT, i);
      d_addr = mpi_comm->getRecvBufferAddr(FRONT, i);
      // src_process = grid_front[i];
      //calculateFrontBoundaryX(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer)
      void *args1[] = {&gauge_, &fermion_out, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr};
      if (i == T_DIRECTION && grid_t > 1) {
        // recv from front
        MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryT, subGridDim, blockDim, args1));
        checkCudaErrors(cudaDeviceSynchronize());
      } else if (i == Z_DIRECTION && grid_z > 1) {
        // recv from front
        MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryZ, subGridDim, blockDim, args1));
        checkCudaErrors(cudaDeviceSynchronize());
      } else if (i == Y_DIRECTION && grid_y > 1) {
        // recv from front
        MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryY, subGridDim, blockDim, args1));
        checkCudaErrors(cudaDeviceSynchronize());
      } else if (i == X_DIRECTION && grid_x > 1) {
        // recv from front
        MPI_Wait(&recv_front_req[i], &recv_front_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateFrontBoundaryX, subGridDim, blockDim, args1));
        checkCudaErrors(cudaDeviceSynchronize());
      }

      h_addr = mpi_comm->getHostRecvBufferAddr(BACK, i);
      d_addr = mpi_comm->getRecvBufferAddr(BACK, i);
      // src_process = grid_back[i];
      // recv from front
      // calculateBackBoundaryT(void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer)
      void *args2[] = {&fermion_out, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr};
      if (i == T_DIRECTION && grid_t > 1) {
        MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryT, subGridDim, blockDim, args2));
        checkCudaErrors(cudaDeviceSynchronize());
      } else if (i == Z_DIRECTION && grid_z > 1) {
        MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryZ, subGridDim, blockDim, args2));
        checkCudaErrors(cudaDeviceSynchronize());
      } else if (i == Y_DIRECTION && grid_y > 1) {
        MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryY, subGridDim, blockDim, args2));
        checkCudaErrors(cudaDeviceSynchronize());
      } else if (i == X_DIRECTION && grid_x > 1) {
        MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
        checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaLaunchKernel((void *)calculateBackBoundaryX, subGridDim, blockDim, args2));
        checkCudaErrors(cudaDeviceSynchronize());
      }

      // recv from back
      // MPI_Wait(&recv_back_req[i], &recv_back_status[i]);
      // checkCudaErrors(cudaMemcpy(d_addr, h_addr, sizeof(Complex) * boundary_length, cudaMemcpyHostToDevice));
      // send to front
      if ((i == T_DIRECTION && grid_t > 1) || 
        (i == Z_DIRECTION && grid_z > 1) || 
        (i == Y_DIRECTION && grid_y > 1) || 
        (i == X_DIRECTION && grid_x > 1) ) {
        MPI_Wait(&send_front_req[i], &send_front_status[i]);
        // send to back
        MPI_Wait(&send_back_req[i], &send_back_status[i]);
      }
    }
    // calc result
  }
  void recvBoundaryVector(int direction) {
    Complex* h_addr;
    int src_process;

    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[direction] = 1;
    int boundary_length = 1;
    int sub_vol = 1;
    for (int i = 0; i < Nd; i++) {
      sub_vol *= length[i];
    }
    sub_vol >>= 1;  // div 2
    boundary_length = sub_vol * (Ns * Nc);

    // from front
    h_addr = mpi_comm->getHostRecvBufferAddr(FRONT, direction);
    src_process = grid_front[direction];
    if ((direction == T_DIRECTION && grid_t > 1) || 
        (direction == Z_DIRECTION && grid_z > 1) || 
        (direction == Y_DIRECTION && grid_y > 1) || 
        (direction == X_DIRECTION && grid_x > 1) )    {
      MPI_Irecv(h_addr, boundary_length * 2, MPI_DOUBLE, src_process, BACK, MPI_COMM_WORLD, &recv_front_req[direction]); // src_process tag is BACK, so use same tag, which is BACK(though from FRONT, so sad)
    }
    // from back
    h_addr = mpi_comm->getHostRecvBufferAddr(BACK, direction);
    src_process = grid_back[direction];
    if ((direction == T_DIRECTION && grid_t > 1) || 
        (direction == Z_DIRECTION && grid_z > 1) || 
        (direction == Y_DIRECTION && grid_y > 1) || 
        (direction == X_DIRECTION && grid_x > 1) )    {
      MPI_Irecv(h_addr, boundary_length * 2, MPI_DOUBLE, src_process, FRONT, MPI_COMM_WORLD, &recv_back_req[direction]);// src_process tag is FRONT, so use same tag, which is FRONT (though from FRONT, so sad)
    }
  }
  void prepareFrontBoundaryVector(void* fermion_in, int direction, int parity) {
    Complex* h_addr;
    Complex* d_addr;

    int dst_process;
    int length[Nd] = {Lx_, Ly_, Lz_, Lt_};
    length[direction] = 1;
    int boundary_length = 1;
    int sub_vol = 1;
    for (int i = 0; i < Nd; i++) {
      sub_vol *= length[i];
    }
    sub_vol >>= 1;  // div 2
    boundary_length = sub_vol * (Ns * Nc);
    dim3 subGridDim(sub_vol / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    // front
    dst_process = grid_front[direction];
    h_addr = mpi_comm->getHostSendBufferAddr(FRONT, direction);
    d_addr = mpi_comm->getSendBufferAddr(FRONT, direction);
    // void DslashTransferFrontT(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer)
    void *args1[] = {&gauge_, &fermion_in, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr};
    if (direction == T_DIRECTION && grid_t > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontT, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
    } else if (direction == Z_DIRECTION && grid_z > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontZ, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
    } else if (direction == Y_DIRECTION && grid_y > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontY, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
    } else if (direction == X_DIRECTION && grid_x > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferFrontX, subGridDim, blockDim, args1));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, FRONT, MPI_COMM_WORLD, &send_front_req[direction]);
    }


    // back
    dst_process = grid_back[direction];
    h_addr = mpi_comm->getHostSendBufferAddr(BACK, direction);
    d_addr = mpi_comm->getSendBufferAddr(BACK, direction);
    void *args2[] = {&fermion_in, &Lx_, &Ly_, &Lz_, &Lt_, &parity, &d_addr};
    // DslashTransferBackT(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer)
    if (direction == T_DIRECTION && grid_t > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackT, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
    } else if (direction == Z_DIRECTION && grid_z > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackZ, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
    } else if (direction == Y_DIRECTION && grid_y > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackY, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
    } else if (direction == X_DIRECTION && grid_x > 1) {
      checkCudaErrors(cudaLaunchKernel((void *)DslashTransferBackX, subGridDim, blockDim, args2));
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(h_addr, d_addr, sizeof(Complex) * boundary_length, cudaMemcpyDeviceToHost));
      MPI_Isend(h_addr, boundary_length * 2, MPI_DOUBLE, dst_process, BACK, MPI_COMM_WORLD, &send_back_req[direction]);
    }

  }
};
__device__ __host__ void gaugeMul (Complex* result, Complex* u1, Complex* u2) {
  Complex temp;
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      temp.clear2Zero();
      for (int k = 0; k < Nc; k++) {
        // result[i * Nc + j] = u1[i * Nc + k] * u2[k * Nc + j];
        temp += u1[i * Nc + k] * u2[k * Nc + j];
      }
      result[i * Nc + j] = temp;
    }
  }
}
__device__ __host__ inline void gaugeAddAssign (Complex* u1, Complex* u2) {
  u1[0] += u2[0];  u1[1] += u2[1];  u1[2] += u2[2];
  u1[3] += u2[3];  u1[4] += u2[4];  u1[5] += u2[5];
  u1[6] += u2[6];  u1[7] += u2[7];  u1[8] += u2[8];
}
__device__ __host__ inline void gaugeSubAssign (Complex* u1, Complex* u2) {
  u1[0] -= u2[0];  u1[1] -= u2[1];  u1[2] -= u2[2];
  u1[3] -= u2[3];  u1[4] -= u2[4];  u1[5] -= u2[5];
  u1[6] -= u2[6];  u1[7] -= u2[7];  u1[8] -= u2[8];
}
__device__ __host__ inline void gaugeAssign(Complex* u1, Complex* u2) {
  u1[0] = u2[0];  u1[1] = u2[1];  u1[2] = u2[2];
  u1[3] = u2[3];  u1[4] = u2[4];  u1[5] = u2[5];
  u1[6] = u2[6];  u1[7] = u2[7];  u1[8] = u2[8];
}
__device__ __host__ inline void gaugeTransposeConj(Complex* u) {
  Complex temp;
  u[0] = u[0].conj(); u[4] = u[4].conj(); u[8] = u[8].conj();  // diag
  temp = u[1];  u[1] = u[3].conj(); u[3] = temp.conj();
  temp = u[2];  u[2] = u[6].conj(); u[6] = temp.conj();
  temp = u[5];  u[5] = u[7].conj(); u[7] = temp.conj();
}

// assume tensor_product is[Ns * Nc * Ns * Nc] and gauge is [Nc * Nc]
__device__ __host__ void tensorProductAddition(Complex* tensor_product, Complex* gauge, int mu, int nu, Complex co_eff = Complex(1, 0)) {
  // 12 times 12 matrix -----> 12 * 6 matrix
  if (mu==0 && nu==1) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==0 && nu==2) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==0 && nu==3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==1 && nu==2) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==1 && nu==3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 + Nc + j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex( 1, 0) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(-1, 0) * gauge[i*Nc + j] * 2;
      }
    }
  } else if (mu==2 && nu==3) {
    for (int i = 0; i < Nc; i++) {
      for (int j = 0; j < Nc; j++) {
        tensor_product[               i*Ns*Nc/2 +      j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
        tensor_product[  Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[2*Nc*Ns*Nc/2 + i*Ns*Nc/2 +      j] += co_eff * Complex(0, -1) * gauge[i*Nc + j] * 2;
        tensor_product[3*Nc*Ns*Nc/2 + i*Ns*Nc/2 + Nc + j] += co_eff * Complex(0,  1) * gauge[i*Nc + j] * 2;
      }
    }
  }
}


// host class
class NeoClover {
private:
  const Complex* origin_gauge_;  // after constructor, never change

  Complex* shift_gauge_1_;
  Complex* shift_gauge_2_;
  Complex* shift_gauge_3_;
public:
  NeoClover(void* origin_gauge) : origin_gauge_(static_cast<Complex*>(origin_gauge)){}
  void gaugeShift(int shift_direction) {
#ifdef QCU_MPI

#else

#endif
  }

  // clear f_buffer to zero, then calc F_1, F_2, F_3, F_4 and add them to buffer
  //  partial_result = 0                   ----F_1-----> partial_result = F_1
  //  partial_result = F_1 + F_2           ----F_2-----> partial_result = F_1 + F_2
  //  partial_result = F_1 + F_2           ----F_3-----> partial_result = F_1 + F_2 + F_3
  //  partial_result = F_1 + F_2 + F_3     ----F_4-----> partial_result = F_1 + F_2 + F_3 + F_4
  // then call cloverCalculate Function, F-F-{\dagger} in cloverCalculate
  void clearResult () {

  }

  void F_1() {}
  void F_2() {}
  void F_3() {}
  void F_4() {}
  void cloverCalculate() {}
};

class Clover {
private:
  Complex* first_item;
  Complex* second_item;
  Complex* third_item;
  Complex* fourth_item;
  Complex* dst_;         // addr to store

  Complex* origin_gauge;
  Complex* shift_gauge[Nd][2];
  Complex* shift_shift_gauge[6][4];
public:
  __device__ int indexTwiceShift(int mu, int nu) {

    assert(mu >= X_DIRECTION && mu <= T_DIRECTION);
    assert(nu >= X_DIRECTION && nu <= T_DIRECTION);
    assert(mu < nu);
    int index = -1;
    switch (mu) {
      case X_DIRECTION:
        if (nu == Y_DIRECTION) index = 0;
        else if (nu == Z_DIRECTION) index = 1;
        else index = 2;
        break;
      case Y_DIRECTION:
        if (nu == Z_DIRECTION) index = 3;
        else index = 4;
        break;
      case Z_DIRECTION:
        index = 5;
        break;
      default:
        break;
    }
    return index;
  }
  __device__ void initialize(Complex* p_origin_gauge, Complex** p_shift_gauge, Complex** p_shift_shift_gauge) {
    origin_gauge = p_origin_gauge;
    for (int i = 0; i < Nd; i++) {
      for (int j = 0; j < 2; j++) {
        shift_gauge[i][j] = p_shift_gauge[i * 2 + j];// mpi_comm->gauge_shift[i][j];
      }
    }
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 4; j++) {
        shift_shift_gauge[i][j] = p_shift_shift_gauge[i * 4 + j];//mpi_comm->gauge_twice_shift[i][j];
      }
    }
  }
  __device__ Clover(Complex* p_origin_gauge, Complex** p_shift_gauge, Complex** p_shift_shift_gauge) {
    initialize(p_origin_gauge, p_shift_gauge, p_shift_shift_gauge);
  }
  __device__ void setDst(Complex* dst) {
    dst_ = dst;
  }
  __device__ Complex* getDst() const{
    return dst_;
  }

  __device__ void F_1(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(origin_gauge, mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(shift_gauge[mu][FRONT], nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(shift_gauge[nu][FRONT], mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(origin_gauge, nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, first_item);
    copyGauge (rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, third_item);
    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, fourth_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ void F_2(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(shift_gauge[mu][BACK], mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(origin_gauge, nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][2], mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(shift_gauge[mu][BACK], nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, second_item);
    copyGauge (rhs, third_item);

    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, fourth_item);
    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ void F_3(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][3], mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(shift_gauge[nu][BACK], nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(shift_gauge[mu][BACK], mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][3], nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, third_item);
    gaugeTransposeConj(lhs);
    copyGauge (rhs, fourth_item);
    gaugeTransposeConj(rhs);

    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);
    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);
  }
  __device__ void F_4(Complex* partial_result, const Point &point, int mu, int nu, int Lx, int Ly, int Lz, int Lt) {
    Complex lhs[Nc * Nc];
    Complex rhs[Nc * Nc];
    first_item = point.getPointGauge(shift_gauge[nu][BACK], mu, Lx, Ly, Lz, Lt);
    second_item = point.getPointGauge(shift_shift_gauge[indexTwiceShift(mu, nu)][1], nu, Lx, Ly, Lz, Lt);
    third_item = point.getPointGauge(origin_gauge, mu, Lx, Ly, Lz, Lt);
    fourth_item = point.getPointGauge(shift_gauge[nu][BACK], nu, Lx, Ly, Lz, Lt);
    copyGauge (lhs, fourth_item);
    gaugeTransposeConj(lhs);
    copyGauge (rhs, first_item);
    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, second_item);
    gaugeMul(partial_result, lhs, rhs);

    gaugeAssign(lhs, partial_result);
    copyGauge (rhs, third_item);
    gaugeTransposeConj(rhs);
    gaugeMul(partial_result, lhs, rhs);
  }

  __device__ void cloverCalculate(Complex* clover_ptr, const Point& point, int Lx, int Ly, int Lz, int Lt) {

    Complex clover_item[Ns * Nc * Ns * Nc / 2];
    Complex temp[Nc * Nc];
    Complex sum_gauge[Nc * Nc];

    setDst(point.getPointClover(clover_ptr, Lx, Ly, Lz, Lt));

    for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
      clover_item[i].clear2Zero();
    }
    for (int i = 0; i < Ns; i++) {  // mu_
      for (int j = i+1; j < Ns; j++) {  //nu_
        for (int jj = 0; jj < Nc * Nc; jj++) {
          sum_gauge[jj].clear2Zero();
        }
        // F_1
        F_1(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        // F_2
        F_2(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        // F_3
        F_3(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        // F_4
        F_4(temp, point, i, j, Lx, Ly, Lz, Lt);
        gaugeAddAssign(sum_gauge, temp);

        gaugeAssign(temp, sum_gauge);
        gaugeTransposeConj(temp);
        gaugeSubAssign(sum_gauge, temp);  // A-A.T.conj()

        tensorProductAddition(clover_item, sum_gauge, i, j, Complex(double(-1)/double(16), 0));
      }
    }

    for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
      dst_[i] = clover_item[i];
    }
  }
};

__global__ void gpuCloverCalculate(void *fermion_out, void* invert_ptr, int Lx, int Ly, int Lz, int Lt, int parity) {
  assert(parity == 0 || parity == 1);
  // __shared__ double dst[BLOCK_SIZE * Ns * Nc * 2];
  __shared__ double shared_buffer[2 * Ns * Nc * BLOCK_SIZE];
  Lx >>= 1;
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;

  Point p(x, y, z, t, parity);
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), Lx, Ly, Lz, Lt);
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex invert_local[Ns * Nc * Ns * Nc / 2];

  // load src vector
  // loadVector(src_local, fermion_out, p, Lx, Ly, Lz, Lt);
  loadVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), src_local);
  // loadCloverBySharedMemory(static_cast<Complex*>(invert_ptr), invert_local);
  Complex* invert_mem = p.getPointClover(static_cast<Complex*>(invert_ptr), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns*Nc*Ns*Nc/2; i++) {
    invert_local[i] = invert_mem[i];
  }

  // A^{-1}dst
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
  for (int i = 0; i < Ns * Nc / 2; i++) {
    for (int j = 0; j < Ns * Nc / 2; j++) {
      dst_local[i] += invert_local[i*Ns*Nc/2+j] * src_local[j];
    }
  }
  for (int i = 0; i < Ns * Nc / 2; i++) {
    for (int j = 0; j < Ns * Nc / 2; j++) {
      dst_local[Ns*Nc/2+i] += invert_local[Ns*Nc*Ns*Nc/4 + i*Ns*Nc/2+j] * src_local[Ns*Nc/2+j];
    }
  }
  // end, and store dst
  // for (int i = 0; i < Ns * Nc; i++) {
  //   dst_ptr[i] = dst_local[i];
  // }
  storeVectorBySharedMemory(static_cast<void*>(shared_buffer), static_cast<Complex*>(fermion_out), dst_local);
}


__global__ void gpuClover(void* clover_ptr, void* invert_ptr, int Lx, int Ly, int Lz, int Lt, int parity, Complex* origin_gauge, Complex** shift_gauge, Complex** shift_shift_gauge) {

  assert(parity == 0 || parity == 1);
  Lx >>= 1;
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * Ly * Lx);
  int z = thread % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread % (Ly * Lx) / Lx;
  int x = thread % Lx;


  Clover clover(origin_gauge, shift_gauge, shift_shift_gauge);

  Point p(x, y, z, t, parity);

  Complex clover_local[(Ns * Nc * Ns * Nc) / 2];
  Complex invert_local[(Ns * Nc * Ns * Nc) / 2];
  Complex* invert_addr = p.getPointClover(static_cast<Complex*>(invert_ptr), Lx, Ly, Lz, Lt);

  clover.cloverCalculate(static_cast<Complex*>(clover_ptr), p, Lx, Ly, Lz, Lt);

  Complex* clover_addr = clover.getDst();


  for (int i = 0; i < Ns * Nc * Ns * Nc / 2; i++) {
    clover_local[i] = clover_addr[i];
  }
  // generate A = 1 + T     TODO: optimize
  for (int i = 0; i < Ns*Nc/2; i++) {
    clover_local[                i*Ns*Nc/2 + i] += Complex(1, 0);
    clover_local[Ns*Nc*Ns*Nc/4 + i*Ns*Nc/2 + i] += Complex(1, 0);
  }
  // store A = 1+T
  // invert A_{oo}
  inverseMatrix(clover_local, invert_local);
  // invert A_{ee}
  inverseMatrix(clover_local + Ns*Nc*Ns*Nc/4, invert_local + Ns*Nc*Ns*Nc/4);
  // store invert vector A_{-1}
  for (int i = 0; i < Ns * Nc * Ns * Nc/2; i++) {
    invert_addr[i] = invert_local[i];
  }

  // storeCloverBySharedMemory(invert_addr, invert_local);
}






// host class
class Dslash {
protected:
  DslashParam *dslashParam_;
public:
  Dslash(DslashParam& param) : dslashParam_(&param){}
  // virtual void calculateDslash() = 0;
  virtual void calculateDslash() = 0;

};


class WilsonDslash : public Dslash {
public:
  WilsonDslash(DslashParam& param) : Dslash(param){}
  virtual void calculateDslash(){//calculateDslash() {
    int Lx = dslashParam_->Lx;
    int Ly = dslashParam_->Ly;
    int Lz = dslashParam_->Lz;
    int Lt = dslashParam_->Lt;
    int parity = dslashParam_->parity;

    int space = Lx * Ly * Lz * Lt >> 1;
    dim3 gridDim(space / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    auto start = std::chrono::high_resolution_clock::now();

    void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity};
    checkCudaErrors(cudaLaunchKernel((void *)gpuDslash, gridDim, blockDim, args));
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
  }
};


// mpiDslash
class MpiWilsonDslash : public Dslash {
public:
  MpiWilsonDslash(DslashParam& param) : Dslash(param){}
  virtual void calculateDslash() {
    int Lx = dslashParam_->Lx;
    int Ly = dslashParam_->Ly;
    int Lz = dslashParam_->Lz;
    int Lt = dslashParam_->Lt;
    int parity = dslashParam_->parity;

    int space = Lx * Ly * Lz * Lt >> 1;
    dim3 gridDim(space / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    mpi_comm->preDslash(dslashParam_->fermion_in, parity);

    auto start = std::chrono::high_resolution_clock::now();
    void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t};
    checkCudaErrors(cudaLaunchKernel((void *)mpiDslash, gridDim, blockDim, args));
    checkCudaErrors(cudaDeviceSynchronize());
    // boundary calculate
    mpi_comm->postDslash(dslashParam_->fermion_out, parity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
    // cloverResult(dslashParam_->fermion_out, invert_matrix, Lx, Ly, Lz, Lt, parity);
  }
  // virtual void calculateDslash(){}
};

static int new_clover_computation;
static bool clover_prepared;
static bool clover_allocated;
static void* clover_matrix;
static void* invert_matrix;

class CloverDslash: public Dslash {
public:
  CloverDslash(DslashParam& param) : Dslash(param){
    int Lx = dslashParam_->Lx;
    int Ly = dslashParam_->Ly;
    int Lz = dslashParam_->Lz;
    int Lt = dslashParam_->Lt;

    int vol = Lx * Ly * Lz * Lt >> 1;
    dim3 gridDim(vol / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    int parity = 0;
    if (!clover_allocated) {
      checkCudaErrors(cudaMalloc(&clover_matrix, sizeof(Complex) * Ns * Nc * Ns * Nc / 2 * Lx * Ly * Lz * Lt));
      checkCudaErrors(cudaMalloc(&invert_matrix, sizeof(Complex) * Ns * Nc * Ns * Nc / 2 * Lx * Ly * Lz * Lt));
      clover_allocated = true;
    }
    if (!clover_prepared) {
      void* origin_gauge = dslashParam_->gauge;
      Complex** shift_gauge = mpi_comm->getShiftGauge();
      Complex** shift_shift_gauge = mpi_comm->getShiftShiftGauge();
      Complex** d_shift_gauge;
      Complex** d_shift_shift_gauge;
      checkCudaErrors(cudaMalloc(&d_shift_gauge, sizeof(Complex) * Nd * 2));
      checkCudaErrors(cudaMalloc(&d_shift_shift_gauge, sizeof(Complex) * 6 * 4));
      checkCudaErrors(cudaMemcpy(d_shift_gauge, shift_gauge, sizeof(Complex) * Nd * 2, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(d_shift_shift_gauge, shift_shift_gauge, sizeof(Complex) * 6 * 4, cudaMemcpyHostToDevice));
      parity = 0;

      void *args[] = {&clover_matrix, &invert_matrix, &Lx, &Ly, &Lz, &Lt, &parity, &origin_gauge, &d_shift_gauge, &d_shift_shift_gauge};
      checkCudaErrors(cudaLaunchKernel((void *)gpuClover, gridDim, blockDim, args));
      checkCudaErrors(cudaDeviceSynchronize());

      parity = 1;
      checkCudaErrors(cudaLaunchKernel((void *)gpuClover, gridDim, blockDim, args));
      checkCudaErrors(cudaDeviceSynchronize());
      
      checkCudaErrors(cudaFree(d_shift_gauge));
      checkCudaErrors(cudaFree(d_shift_shift_gauge));
      if (new_clover_computation) {
        clover_prepared = false;
      } else {
        clover_prepared = true;
      }
    }
  }
  void cloverResult(void* p_fermion_out, void* p_invert_matrix, int Lx, int Ly, int Lz, int Lt, int parity) {
    int space = Lx * Ly * Lz * Lt >> 1;
    dim3 gridDim(space / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    void *args[] = {&p_fermion_out, &p_invert_matrix, &Lx, &Ly, &Lz, &Lt, &parity};
    checkCudaErrors(cudaLaunchKernel((void *)gpuCloverCalculate, gridDim, blockDim, args));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  virtual void calculateDslash() {
    int Lx = dslashParam_->Lx;
    int Ly = dslashParam_->Ly;
    int Lz = dslashParam_->Lz;
    int Lt = dslashParam_->Lt;
    int parity = dslashParam_->parity;

    int space = Lx * Ly * Lz * Lt >> 1;
    dim3 gridDim(space / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    mpi_comm->preDslash(dslashParam_->fermion_in, parity);

    auto start = std::chrono::high_resolution_clock::now();
    void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t};
    checkCudaErrors(cudaLaunchKernel((void *)mpiDslash, gridDim, blockDim, args));
    checkCudaErrors(cudaDeviceSynchronize());
    // boundary calculate
    mpi_comm->postDslash(dslashParam_->fermion_out, parity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("total time: (without malloc free memcpy) : %.9lf sec\n", double(duration) / 1e9);
    cloverResult(dslashParam_->fermion_out, invert_matrix, Lx, Ly, Lz, Lt, parity);
  }
};

__attribute__((constructor)) void initialize() {
  mpi_comm = nullptr;
  clover_prepared = false;
  new_clover_computation = 0;
  clover_allocated = false;
  // static cudaStream_t stream[Nd][2]; 
  // for (int i = 0; i < Nd; i++) {
  //   for (int j = 0; j < 2; j++) {
  //     cudaStreamCreate(&stream[i][j]);
  //   } // end for j
  // } // end for i
}
__attribute__((destructor)) void destroySpace() {
  delete mpi_comm;
  // for (int i = 0; i < Nd; i++) {
  //   for (int j = 0; j < 2; j++) {
  //     cudaStreamDestroy(stream[i][j]);
  //   } // end for j
  // } // end for i
}

void initMPICommunicator(void* gauge, void* fermion_in, void* fermion_out, int Lx, int Ly, int Lz, int Lt) {
  mpi_comm = new MPICommunicator(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt);
}

// initialize the struct of grid, first function to call
void initGridSize(QcuGrid_t* grid, QcuParam* p_param, void* gauge, void* fermion_in, void* fermion_out) {
  // x,y,z,t
  int Lx = p_param->lattice_size[0];
  int Ly = p_param->lattice_size[1];
  int Lz = p_param->lattice_size[2];
  int Lt = p_param->lattice_size[3];

  grid_x = grid->grid_size[0];
  grid_y = grid->grid_size[1];
  grid_z = grid->grid_size[2];
  grid_t = grid->grid_size[3];

  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_num);
  coord.t = process_rank % grid_t;
  coord.z = process_rank / grid_t % grid_z;
  coord.y = process_rank / grid_t / grid_z % grid_y;
  coord.x = process_rank / grid_t / grid_z / grid_y;  // rank divide(g_y*g_z*g_t)
  // mpi_comm = new MPICommunicator(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt);
  initMPICommunicator(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt);
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {

  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  // MpiWilsonDslash dslash_solver(dslash_param);
  CloverDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash();
}
