#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "qcu.h"
#include <cuda_runtime.h>
#include "qcu_complex.cuh"
#include "qcu_complex_computation.cuh"
#include "qcu_macro.cuh"
#include "qcu_clover_dslash.cuh"
#include "qcu_communicator.cuh"
// #define DEBUG

extern MPICommunicator *mpi_comm;




// modify b
void generate_new_b(void* new_b, void* origin_b, void* gauge, QcuParam *param, double kappa = 1.0) {
  // be + bo ----> be + new bo

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  void* new_even_b = new_b;
  void* new_odd_b = static_cast<Complex*>(new_b) + half_vol * Ns * Nc;

  // void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);
  // callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);


}

// void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);
void matrix_mul_vector_odd(void* full_out, void* full_in, void* gauge, QcuParam *param, int parity, int dagger_flag, double kappa = 1.0) {
  void* real_in;
  void* real_out;
}


void cg_inverter(void* x_vector, void* b_vector, void *gauge, QcuParam *param) {
  int total_vol = param->lattice_size[0] * param->lattice_size[1] * param->lattice_size[2] * param->lattice_size[3];
  int half_vol = total_vol >> 1;
  void* temp_vec1;


  checkCudaErrors(cudaMalloc(&temp_vec1, sizeof(Complex) * total_vol));





  checkCudaErrors(cudaFree(temp_vec1));
}