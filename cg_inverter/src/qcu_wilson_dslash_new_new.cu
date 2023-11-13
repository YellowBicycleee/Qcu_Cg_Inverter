// #include "qcu_wilson_dslash_neo.cuh"
#include "qcu_complex.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
// #include "qcu_shift_storage.cuh"
#include <chrono>
#include "qcu_wilson_dslash_new_new.cuh"
#define INCLUDE_COMPUTATION




extern int grid_x;
extern int grid_y;
extern int grid_z;
extern int grid_t;
extern MPICommunicator *mpi_comm;

// static void* coalesced_fermion_in;
// static void* coalesced_fermion_out;
// static void* coalesced_gauge;
// extern void* qcu_gauge;




static __device__ __forceinline__ void reconstructSU3(double *su3)
{
  // su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  // su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  // su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
  su3[6 * 2 + 0] = (su3[1 * 2 + 0] * su3[5 * 2 + 0] - su3[1 * 2 + 1] * su3[5 * 2 + 1]) \
                 - (su3[2 * 2 + 0] * su3[4 * 2 + 0] - su3[2 * 2 + 1] * su3[4 * 2 + 1]);
  su3[6 * 2 + 1] = (su3[2 * 2 + 1] * su3[4 * 2 + 0] + su3[2 * 2 + 0] * su3[4 * 2 + 1]) \
                 - (su3[1 * 2 + 1] * su3[5 * 2 + 0] + su3[1 * 2 + 0] * su3[5 * 2 + 1]); // conj()

  su3[7 * 2 + 0] = (su3[2 * 2 + 0] * su3[3 * 2 + 0] - su3[2 * 2 + 1] * su3[3 * 2 + 1]) \
                 - (su3[0 * 2 + 0] * su3[5 * 2 + 0] - su3[0 * 2 + 1] * su3[5 * 2 + 1]);
  su3[7 * 2 + 1] = (su3[0 * 2 + 1] * su3[5 * 2 + 0] + su3[0 * 2 + 0] * su3[5 * 2 + 1]) \
                 - (su3[2 * 2 + 1] * su3[3 * 2 + 0] + su3[2 * 2 + 0] * su3[3 * 2 + 1]); // conj()

  su3[8 * 2 + 0] = (su3[0 * 2 + 0] * su3[4 * 2 + 0] - su3[0 * 2 + 1] * su3[4 * 2 + 1]) \
                 - (su3[1 * 2 + 0] * su3[3 * 2 + 0] - su3[1 * 2 + 1] * su3[3 * 2 + 1]);
  su3[8 * 2 + 1] = (su3[1 * 2 + 1] * su3[3 * 2 + 0] + su3[1 * 2 + 0] * su3[3 * 2 + 1]) \
                 - (su3[0 * 2 + 1] * su3[4 * 2 + 0] + su3[0 * 2 + 0] * su3[4 * 2 + 1]); // conj()
}



__device__ __forceinline__ void loadGauge(double* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  double* u = reinterpret_cast<double*>(p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt));
  for (int i = 0; i < (Nc - 1) * Nc * 2; i++) {
    u_local[i] = u[i];
  }
  // reconstructSU3(u_local);
  reconstructSU3(reinterpret_cast<double*>(u_local));
}





__device__ __forceinline__ void loadVector(double* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  // Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  double* src_double = reinterpret_cast<double*>(p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt));

  for (int i = 0; i < Ns * Nc * 2; i++) {
    src_local[i] = src_double[i];
  }
}



static __global__ void mpiDslashNew(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, double flag_param) {
  assert(parity == 0 || parity == 1);
  // __shared__ double shared_buffer[BLOCK_SIZE * Ns * Nc * 2];

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
  double u_local[Nc * Nc * 2];   // for GPU
  double src_local[Ns * Nc * 2]; // for GPU
  double dst_local[Ns * Nc * 2]; // for GPU
  // Complex u_local[Nc * Nc];   // for GPU
  // Complex src_local[Ns * Nc]; // for GPU
  // Complex dst_local[Ns * Nc]; // for GPU

  // double* u_local_double_ptr = reinterpret_cast<double*>(u_local);
  // double* src_local_double_ptr = reinterpret_cast<double*>(src_local);
  // double* dst_local_double_ptr = reinterpret_cast<double*>(dst_local);
  // Complex temp;
  // Complex temp1;
  // Complex temp2;

  double temp_reg[2];
  double temp_res1[2];
  double temp_res2[2];
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc * 2; i++) {
    // dst_local[i].clear2Zero();
    dst_local[i] = 0;
  }

  // \mu = 1
  loadGauge(u_local, gauge, X_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {

    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    - (-src_local[(3 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    - (src_local[(3 * Nc + j) * 2 + 0] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res1[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];

        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    - (-src_local[(2 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    - (src_local[(2 * Nc + j) * 2 + 0] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res2[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];
      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[3 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(3 * Nc + i) * 2 + 0] += -flag * temp_res1[1];
      dst_local[(3 * Nc + i) * 2 + 1] += flag * temp_res1[0];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[2 * Nc + i] += temp2.multipy_i() * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(2 * Nc + i) * 2 + 0] += -flag * temp_res2[1];
      dst_local[(2 * Nc + i) * 2 + 1] += flag * temp_res2[0];
    }

  }
  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);;

  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) *
        //       u_local[j * Nc + i].conj(); // transpose and conj

        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    + (-src_local[(3 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    + (src_local[(3 * Nc + j) * 2 + 0] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res1[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];


        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
        //       u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    + (-src_local[(2 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    + (src_local[(2 * Nc + j) * 2 + 0] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res2[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];
      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[3 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(3 * Nc + i) * 2 + 0] += flag * temp_res1[1];
      dst_local[(3 * Nc + i) * 2 + 1] += -flag * temp_res1[0];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[2 * Nc + i] += temp2.multipy_minus_i() * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(2 * Nc + i) * 2 + 0] += flag * temp_res2[1];
      dst_local[(2 * Nc + i) * 2 + 1] += -flag * temp_res2[0];
    }
  }

  // \mu = 2
  // y front
  loadGauge(u_local, gauge, Y_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    + (src_local[(3 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    + (src_local[(3 * Nc + j) * 2 + 1] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res1[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];


        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j] *  flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    - (src_local[(2 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    - (src_local[(2 * Nc + j) * 2 + 1] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res2[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];

      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[3 * Nc + i] += temp1 * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(3 * Nc + i) * 2 + 0] += flag * temp_res1[0];
      dst_local[(3 * Nc + i) * 2 + 1] += flag * temp_res1[1];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[2 * Nc + i] += -temp2 * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(2 * Nc + i) * 2 + 0] += -flag * temp_res2[0];
      dst_local[(2 * Nc + i) * 2 + 1] += -flag * temp_res2[1];
    }
  }

  // y back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    - (src_local[(3 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    - (src_local[(3 * Nc + j) * 2 + 1] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res1[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];

        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    + (src_local[(2 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    + (src_local[(2 * Nc + j) * 2 + 1] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res2[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];

      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[3 * Nc + i] += -temp1 * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(3 * Nc + i) * 2 + 0] += -flag * temp_res1[0];
      dst_local[(3 * Nc + i) * 2 + 1] += -flag * temp_res1[1];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[2 * Nc + i] += temp2 * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(2 * Nc + i) * 2 + 0] += flag * temp_res2[0];
      dst_local[(2 * Nc + i) * 2 + 1] += flag * temp_res2[1];
    }
  }

  // \mu = 3
  // z front
  loadGauge(u_local, gauge, Z_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    - (-src_local[(2 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    - (src_local[(2 * Nc + j) * 2 + 0] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res1[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];

        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    + (-src_local[(3 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    + (src_local[(3 * Nc + j) * 2 + 0] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res2[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];
      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[2 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(2 * Nc + i) * 2 + 0] += -flag * temp_res1[1];
      dst_local[(2 * Nc + i) * 2 + 1] += flag * temp_res1[0];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[3 * Nc + i] += temp2.multipy_minus_i() * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(3 * Nc + i) * 2 + 0] += flag * temp_res2[1];
      dst_local[(3 * Nc + i) * 2 + 1] += -flag * temp_res2[0];
    }
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              // u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    + (-src_local[(2 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    + (src_local[(2 * Nc + j) * 2 + 0] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res1[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];


        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) *
        //       u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    - (-src_local[(3 * Nc + j) * 2 + 1] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    - (src_local[(3 * Nc + j) * 2 + 0] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res2[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];
      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[2 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(2 * Nc + i) * 2 + 0] += flag * temp_res1[1];
      dst_local[(2 * Nc + i) * 2 + 1] += -flag * temp_res1[0];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[3 * Nc + i] += temp2.multipy_i() * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(3 * Nc + i) * 2 + 0] += -flag * temp_res2[1];
      dst_local[(3 * Nc + i) * 2 + 1] += flag * temp_res2[0];
    }
  }

  // t: front
  // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j] * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    - (src_local[(2 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    - (src_local[(2 * Nc + j) * 2 + 1] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res1[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];
        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    - (src_local[(3 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    - (src_local[(3 * Nc + j) * 2 + 1] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 0] \
                      - temp_reg[1] * u_local[(i * Nc + j) * 2 + 1];
        temp_res2[1] += temp_reg[0] * u_local[(i * Nc + j) * 2 + 1] \
                      + temp_reg[1] * u_local[(i * Nc + j) * 2 + 0];
      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[2 * Nc + i] += -temp1 * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(2 * Nc + i) * 2 + 0] += -flag * temp_res1[0];
      dst_local[(2 * Nc + i) * 2 + 1] += -flag * temp_res1[1];

      // dst_local[1 * Nc + i] += temp2;
      // dst_local[3 * Nc + i] += -temp2 * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(3 * Nc + i) * 2 + 0] += -flag * temp_res2[0];
      dst_local[(3 * Nc + i) * 2 + 1] += -flag * temp_res2[1];
    }
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(reinterpret_cast<double*>(u_local), gauge, T_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(reinterpret_cast<double*>(src_local), fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
    #pragma unroll
    for (int i = 0; i < Nc; i++) {
      // temp1.clear2Zero();
      // temp2.clear2Zero();
      temp_res1[0] = temp_res1[1] = 0;
      temp_res2[0] = temp_res2[1] = 0;
      #pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        // temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(0 * Nc + j) * 2 + 0] \
                    + (src_local[(2 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(0 * Nc + j) * 2 + 1] \
                    + (src_local[(2 * Nc + j) * 2 + 1] * flag));

        temp_res1[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res1[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];

        // second row vector with col vector
        // temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        temp_reg[0] = (src_local[(1 * Nc + j) * 2 + 0] \
                    + (src_local[(3 * Nc + j) * 2 + 0] * flag));
        temp_reg[1] = (src_local[(1 * Nc + j) * 2 + 1] \
                    + (src_local[(3 * Nc + j) * 2 + 1] * flag));

        temp_res2[0] += temp_reg[0] * u_local[(j * Nc + i) * 2 + 0] \
                      - temp_reg[1] * (-u_local[(j * Nc + i) * 2 + 1]);
        temp_res2[1] += temp_reg[0] * (-u_local[(j * Nc + i) * 2 + 1]) \
                      + temp_reg[1] * u_local[(j * Nc + i) * 2 + 0];
      }
      // dst_local[0 * Nc + i] += temp1;
      // dst_local[2 * Nc + i] += temp1 * flag;
      dst_local[(0 * Nc + i) * 2 + 0] += temp_res1[0];
      dst_local[(0 * Nc + i) * 2 + 1] += temp_res1[1];
      dst_local[(2 * Nc + i) * 2 + 0] += flag * temp_res1[0];
      dst_local[(2 * Nc + i) * 2 + 1] += flag * temp_res1[1];
  
      // dst_local[1 * Nc + i] += temp2;
      // dst_local[3 * Nc + i] += temp2 * flag;
      dst_local[(1 * Nc + i) * 2 + 0] += temp_res2[0];
      dst_local[(1 * Nc + i) * 2 + 1] += temp_res2[1];
      dst_local[(3 * Nc + i) * 2 + 0] += flag * temp_res2[0];
      dst_local[(3 * Nc + i) * 2 + 1] += flag * temp_res2[1];
    }
  }

  double* dst_global = reinterpret_cast<double*>(p.getPointVector(static_cast<Complex *>(fermion_out), Lx, Ly, Lz, Lt));
  for (int i = 0; i < Ns * Nc * 2; i++) {
    dst_global[i] = dst_local[i];
  }
}




static __device__ __forceinline__ void loadGaugeCoalesced(double* u_local, void* gauge_ptr, int direction, const Point& p, int sub_Lx, int Ly, int Lz, int Lt) {
  double* start_ptr = reinterpret_cast<double*>(p.getCoalescedGaugeAddr (gauge_ptr, direction, sub_Lx, Ly, Lz, Lt));
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[2*i] = start_ptr[0];
    u_local[2*i+1] = start_ptr[1];
    start_ptr += sub_vol * 2;
  }
  reconstructSU3(u_local);
}

static __device__ __forceinline__ void loadVectorCoalesced(double* src_local, void* fermion_in, const Point& p, int half_Lx, int Ly, int Lz, int Lt) {
  // Complex* start_ptr = p.getCoalescedVectorAddr (fermion_in, half_Lx, Ly, Lz, Lt);
  double* start_ptr = reinterpret_cast<double*>(p.getCoalescedVectorAddr (fermion_in, half_Lx, Ly, Lz, Lt));

  int sub_vol = half_Lx * Ly * Lz * Lt;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[2*i] = start_ptr[0];
    src_local[2*i+1] = start_ptr[1];
    start_ptr += sub_vol * 2;
  }
}

static __device__ __forceinline__ void storeVectorCoalesced(double* dst_local, void* fermion_out, const Point& p, int half_Lx, int Ly, int Lz, int Lt) {
  // Complex* start_ptr = p.getCoalescedVectorAddr (fermion_out, half_Lx, Ly, Lz, Lt);
  double* start_ptr = reinterpret_cast<double*>(p.getCoalescedVectorAddr (fermion_out, half_Lx, Ly, Lz, Lt));

  int sub_vol = half_Lx * Ly * Lz * Lt;

  for (int i = 0; i < Ns * Nc; i++) {
    start_ptr[0] = dst_local[2*i+0];
    start_ptr[0] = dst_local[2*i+1];
    start_ptr += sub_vol * 2;
  }
}


void NewDslash::calculateDslash(int invert_flag) {
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

  checkCudaErrors(cudaLaunchKernel((void *)mpiDslashNew, gridDim, blockDim, args));
  checkCudaErrors(cudaDeviceSynchronize());

  // boundary calculate
  mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("naive without class total time: (without malloc free memcpy) : %.9lf sec, block size = %d\n", double(duration) / 1e9, BLOCK_SIZE);
}


void callNewDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag) {
  DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
  NewDslash dslash_solver(dslash_param);
  dslash_solver.calculateDslash(0);
}
