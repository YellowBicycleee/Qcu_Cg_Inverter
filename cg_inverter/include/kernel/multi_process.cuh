#pragma once

#include "qcu_complex.cuh"
#include "qcu_point.cuh"
// #include "kernel/public_kernels.cuh"

// static __device__ __forceinline__ void reconstructSU3(Complex *su3


static __device__ __forceinline__ void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}
static __device__ __forceinline__ void copyGauge (Complex* dst, Complex* src) {
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    dst[i] = src[i];
  }
  reconstructSU3(dst);
}

static __device__ __forceinline__ void loadGauge(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* u = p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
static __device__ __forceinline__ void loadVector(Complex* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}

static __device__ __forceinline__ void storeVector(Complex* src_local, void* fermion_out, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_out), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src[i] = src_local[i];
  }
}

__global__ void DslashTransferFrontX(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int sub_Ly = (Ly >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Ly);
  int z = thread % (Lz * sub_Ly) / sub_Ly;
  int sub_y = thread % sub_Ly;
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  int new_even_odd = (z+t) & 0x01;
  Point p(sub_Lx-1, 2 * sub_y + (new_even_odd == 1-parity), z, t, 1-parity);
  Point dst_p(0, sub_y, z, t, 0); // parity is useless
  // Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  // even save to even, odd save to even
  // dst_ptr = send_buffer + thread*Ns*Nc;  

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + flag * src_local[3 * Nc + j] * Complex(0, 1)) *
            u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1) * flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + flag * src_local[2 * Nc + j] * Complex(0, 1)) *
            u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1) * flag;
    }
  }

  // for (int i = 0; i < Ns * Nc; i++) {
  //   dst_ptr[i] = dst_local[i];
  // }
  // x轴与其他轴不同
  storeVector(dst_local, send_buffer, dst_p, 1, sub_Ly, Lz, Lt);
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
  Point dst_p(0, sub_y, z, t, 0); // parity is useless
  // Complex* dst_ptr;
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);

  // dst_ptr = send_buffer + thread * Ns * Nc;
  // for (int i = 0; i < Ns * Nc; i++) {
  //   dst_ptr[i] = src_local[i];
  // }
  storeVector(src_local, send_buffer, dst_p, 1, sub_Ly, Lz, Lt);
}
__global__ void DslashTransferFrontY(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, Ly-1, z, t, 1-parity);
  Point dst_p(x, 0, z, t, 0); // parity is useless
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, p, sub_Lx, Ly, Lz, Lt);


  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - flag * src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp * flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + flag * src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * flag;
    }
  }

  storeVector(dst_local, send_buffer, dst_p, sub_Lx, 1, Lz, Lt);
}
__global__ void DslashTransferBackY(void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer) {
  // 后传传向量
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Lz * sub_Lx);
  int z = thread % (Lz * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, 0, z, t, 1-parity);
  Point dst_p(x, 0, z, t, 0); // parity is useless
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  storeVector(src_local, send_buffer, dst_p, sub_Lx, 1, Lz, Lt);
}
// DslashTransferFrontZ: DONE
__global__ void DslashTransferFrontZ(void *gauge, void *fermion_in,int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr) {
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, Lz-1, t, 1-parity);
  Point dst_p(x, y, 0, t, 0); // parity is useless
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex* dst_ptr;

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, p, sub_Lx, Ly, Lz, Lt);


  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + flag * src_local[2 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, -1) * flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - flag * src_local[3 * Nc + j] * Complex(0, 1)) *
             u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1) * flag;
    }
  }

  storeVector(dst_local, send_buffer, dst_p, sub_Lx, Ly, 1, Lt);
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
  Point dst_p(x, y, 0, t, 0);
  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  storeVector(src_local, send_buffer, dst_p, sub_Lx, 1, Lz, Lt);
}

// DslashTransferFrontT: Done
__global__ void DslashTransferFrontT(void *gauge, void *fermion_in, int Lx, int Ly, int Lz, int Lt, int parity, Complex* send_buffer, void* flag_ptr){
  // 前传传结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt-1, 1-parity);
  Point dst_p(x, y, z, 0, 0); // parity is useless
  Complex flag = *(static_cast<Complex*>(flag_ptr));

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];
  Complex temp;
  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, sub_Lx, Ly, Lz, Lt);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }
  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] + flag * src_local[2 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * flag;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + flag * src_local[3 * Nc + j]) * u_local[j * Nc + i].conj(); // transpose and conj
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * flag;
    }
  }

  storeVector(dst_local, send_buffer, dst_p, sub_Lx, Ly, Lz, 1);
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
  Point dst_p(x, y, z, 0, 0); // parity is useless

  Complex src_local[Ns * Nc];

  loadVector(src_local, fermion_in, p, sub_Lx, Ly, Lz, Lt);
  storeVector(src_local, send_buffer, dst_p, sub_Lx, Ly, Lz, 1);
}

// ---separate line-----
// after this is postDslash kernels

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
__global__ void calculateFrontBoundaryX(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, double dagger_flag_double) {
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
      temp = (src_local[0 * Nc + j] - src_local[3 * Nc + j] * dagger_flag_double * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, 1) * dagger_flag_double;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * dagger_flag_double * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1) * dagger_flag_double;
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

__global__ void calculateFrontBoundaryY(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, double dagger_flag_double) {
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
      temp = (src_local[0 * Nc + j] + src_local[3 * Nc + j] * dagger_flag_double) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * dagger_flag_double;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[2 * Nc + j] * dagger_flag_double) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp * dagger_flag_double;
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

__global__ void calculateFrontBoundaryZ(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, double dagger_flag_double) {
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
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * dagger_flag_double * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += temp * Complex(0, 1) * dagger_flag_double;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] + src_local[3 * Nc + j] * dagger_flag_double * Complex(0, 1)) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += temp * Complex(0, -1) * dagger_flag_double;
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
  Point buffer_p(x, y, z, 0, 0); // parity is useless
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, Ly, Lz, 1);

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}

__global__ void calculateFrontBoundaryT(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* recv_buffer, double dagger_flag_double) {
  // 后接接结果
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;
  Point p(x, y, z, Lt-1, parity);
  Point buffer_p(x, y, z, 0, 0); // parity is useless
  Complex temp;
  Complex* dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  Complex src_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex dst_local[Ns * Nc];

  loadGauge(u_local, gauge, 3, p, sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
  loadVector(src_local, recv_buffer, buffer_p, sub_Lx, Ly, Lz, 1);

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * dagger_flag_double) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp * dagger_flag_double;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * dagger_flag_double) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp * dagger_flag_double;
    }
  }

  storeVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);
}


// combine front and back : T
__global__ void calculateBoundaryT(void* gauge, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, Complex* back_buffer, Complex* front_buffer, double dagger_flag_double) {
  // back
  int sub_Lx = (Lx >> 1);
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int z = thread / (Ly * sub_Lx);
  int y = thread % (Ly * sub_Lx) / sub_Lx;
  int x = thread % sub_Lx;

  Point p;
  Complex* src_ptr;
  Complex* dst_ptr;
  Complex src_local[Ns * Nc];
  Complex dst_local[Ns * Nc];
  Complex u_local[Nc * Nc];
  Complex temp;

  p = Point(x, y, z, 0, parity);

  dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);
  loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  // load back buffer
  src_ptr = back_buffer + thread * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i] += src_local[i];
  }

  // front
  p = Point(x, y, z, Lt-1, parity);
  dst_ptr = p.getPointVector(static_cast<Complex*>(fermion_out), sub_Lx, Ly, Lz, Lt);

  loadGauge(u_local, gauge, T_DIRECTION, p, sub_Lx, Ly, Lz, Lt);
  // loadVector(dst_local, fermion_out, p, sub_Lx, Ly, Lz, Lt);

  src_ptr = front_buffer + thread * Ns * Nc;//((z * Ly + y) * sub_Lx + x) * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src_ptr[i];
  }
  // save back result
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      // first row vector with col vector
      temp = (src_local[0 * Nc + j] - src_local[2 * Nc + j] * dagger_flag_double) * u_local[i * Nc + j];
      dst_local[0 * Nc + i] += temp;
      dst_local[2 * Nc + i] += -temp * dagger_flag_double;
      // second row vector with col vector
      temp = (src_local[1 * Nc + j] - src_local[3 * Nc + j] * dagger_flag_double) * u_local[i * Nc + j];
      dst_local[1 * Nc + i] += temp;
      dst_local[3 * Nc + i] += -temp * dagger_flag_double;
    }
  }
  // save front result
  for (int i = 0; i < Ns * Nc; i++) {
    dst_ptr[i] = dst_local[i];
  }
}