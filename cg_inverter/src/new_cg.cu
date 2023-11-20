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
#include "qcu_wilson_dslash_neo.cuh"

#define DEBUG

extern MPICommunicator *mpi_comm;

static __global__ void clearVectorKernel(void* vec, int vector_length) {
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  Complex* src = static_cast<Complex*>(vec);
  if (thread_id >= vector_length) {
    return;
  }
  int vol = blockDim.x * gridDim.x;

  for (int i = thread_id; i < vol * Ns * Nc; i += vol) {
    src[i].clear2Zero();
  }

}

static void clear_vector (void* vec, int vector_length) {
  int block_size = MAX_BLOCK_SIZE;
  int grid_size = (vector_length + block_size * Ns * Nc - 1) / (block_size * Ns * Nc);
  clearVectorKernel<<<grid_size, block_size>>>(vec, vector_length);
  qcuCudaDeviceSynchronize();
}







// iter_result_odd, iter_b_odd --->half vol
// temp_vec1, temp_vec2---->half vol
void odd_solver_dslash_dagger (void* output_Ax, void* input_x, void* temp_vec1, void* temp_vec2, void* gauge, void* d_kappa, QcuParam *param, double kappa = 1.0) {

  Complex h_coeff;
  int parity;
  int dagger_flag;

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;


  qcuCudaMemcpy(output_Ax, input_x, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice);


  parity = 1;
  // part 1
  cloverVectorHalf (output_Ax, nullptr, gauge, param, parity);


  // part 2
  parity = 0;
  dagger_flag = 1;
  callWilsonDslashNaive(temp_vec1, input_x, gauge, param, parity, dagger_flag);

  // clover invert
  parity = 0;
  invertCloverDslashHalf (temp_vec1, nullptr, gauge, param, parity);


  parity = 1;
  callWilsonDslashNaive(temp_vec2, temp_vec1, gauge, param, parity, dagger_flag);


  h_coeff = Complex(-kappa * kappa, 0);
  qcuCudaMemcpy(d_kappa, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));


  // saxpy
  mpi_comm->interprocess_saxpy_barrier(temp_vec2, output_Ax, d_kappa, half_vol);  // coeff temp2 + x --->x
}




void odd_solver_dslash (void* output_Ax, void* input_x, void* temp_vec1, void* temp_vec2, void* gauge, void* d_kappa, QcuParam *param, double kappa = 1.0) {

  Complex h_coeff;
  int parity;
  int dagger_flag;

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];


  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;


  qcuCudaMemcpy(output_Ax, input_x, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice));
  parity = 1;
  // part 1
  cloverVectorHalf (output_Ax, nullptr, gauge, param, parity);

  // part 2
  parity = 0;
  dagger_flag = 0;
  callWilsonDslashNaive(temp_vec1, input_x, gauge, param, parity, dagger_flag);
  // clover invert
  parity = 0;
  invertCloverDslashHalf (temp_vec1, nullptr, gauge, param, parity);
  parity = 1;
  callWilsonDslashNaive(temp_vec2, temp_vec1, gauge, param, parity, dagger_flag);

  h_coeff = Complex(-kappa * kappa, 0);
  checkCudaErrors(cudaMemcpy(d_kappa, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));

  // saxpy
  mpi_comm->interprocess_saxpy_barrier(temp_vec2, output_Ax, d_kappa, half_vol);  // coeff temp2 + x --->x
}

void full_odd_solver_dslash (void* output_Ax, void* input_x, void* temp_vec1, void* temp_vec2, void* temp_vec3, void* gauge, void* d_kappa, QcuParam *param, double kappa = 1.0) {
  odd_solver_dslash (temp_vec3, input_x, temp_vec1, temp_vec2, gauge, d_kappa, param, kappa);
  odd_solver_dslash_dagger (output_Ax, temp_vec3, temp_vec1, temp_vec2, gauge, d_kappa, param, kappa);
}


// current_b is temporary
bool if_odd_converge(void* current_x, void* current_b_buffer, void* target_b, void* temp_vec1, void* temp_vec2, void* temp_vec3, void* gauge, void* d_kappa, void* d_coeff, void* d_norm1, void* d_norm2, QcuParam *param, double kappa = 1.0) {
  //   if (if_odd_converge(iter_x_odd, temp_vec5, target_b, temp_vec2, temp_vec3, temp_vec4, gauge, d_kappa, d_coeff, d_norm1, d_norm2, param, kappa))
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  Complex h_coeff;


  full_odd_solver_dslash (current_b_buffer, current_x, temp_vec1, temp_vec2, temp_vec3, gauge, d_kappa, param, kappa);

  double h_norm1; // norm(b)
  double h_norm2; // norm(thisb -b)


  gpu_vector_norm2(target_b, temp_vec3, half_vol, d_norm1);

  checkCudaErrors(cudaMemcpy(temp_vec2, target_b, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice));     // target_b -----> temp_vec2
  h_coeff = Complex(-1, 0);
  checkCudaErrors(cudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(current_b_buffer, temp_vec2, d_coeff, half_vol); // temp_vec2 <--- target_b - current_b

  gpu_vector_norm2(temp_vec2, temp_vec3, half_vol, d_norm2);
  checkCudaErrors(cudaMemcpy(&h_norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&h_norm2, d_norm2, sizeof(double), cudaMemcpyDeviceToHost));
  printf("difference %lf, norm1 = %lf, norm2 = %lf\n", h_norm2 / h_norm1, h_norm1, h_norm2);
  if (h_norm2 / h_norm1 < 1e-23) {
    return true;
  }
  else {
    return false;
  }
}

bool odd_cg_iter(void* iter_x_odd, void* target_b, void* resid_vec, void* p_vec, \
        void* temp_vec1, void* temp_vec2, void* temp_vec3, void* temp_vec4, void* temp_vec5, \
        void* gauge, QcuParam *param, double kappa, void* d_kappa, \
        void* d_alpha, void* d_beta, void* d_denominator, void* d_numerator, void* d_coeff, void* d_norm1, void* d_norm2) {
  int vol = param->lattice_size[0] * param->lattice_size[1] * param->lattice_size[2] *param->lattice_size[3];
  int half_vol = vol >> 1;
  // bool if_end = false;

  Complex alpha;
  Complex beta;
  Complex denominator;
  Complex numerator;
  Complex one(1,0);

  // <r, r>--->denominator
  mpi_comm->interprocess_inner_prod_barrier(resid_vec, resid_vec, d_numerator, half_vol);  // <r, r> --> d_numerator

  full_odd_solver_dslash (temp_vec4, p_vec, temp_vec1, temp_vec2, temp_vec3, gauge, d_kappa, param, kappa); // Ap --->temp_vec4


  mpi_comm->interprocess_inner_prod_barrier(p_vec, temp_vec4, d_denominator, half_vol);  // <p, Ap> --> d_numerator

  checkCudaErrors(cudaMemcpy(&numerator, d_numerator, sizeof(Complex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost));
  alpha = numerator / denominator;
  checkCudaErrors(cudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice));

  mpi_comm->interprocess_saxpy_barrier(p_vec, iter_x_odd, d_alpha, half_vol); // x = x + \alpha p

  checkCudaErrors(cudaMemcpy(temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice)); // copy r to temp_vec1  r'=r
// #ifdef DEBUG
//   double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(temp_vec1, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of temp_vec1 is %lf\n", norm);
//   gpu_vector_norm2(temp_vec4, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of temp_vec4 is %lf\n", norm);
//   printf("alhpa.real = %lf, alhpa.imag = %lf\n", alpha.real(), alpha.imag());

//   printf(CLR"");
// #endif

  alpha = alpha * Complex(-1, 0);
  checkCudaErrors(cudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(temp_vec4, temp_vec1, d_alpha, half_vol); // temp_vec4 = Ap, r'=r'-\alpha Ap------>temp_vec1

// #ifdef DEBUG
//   // double norm;
//   printf(BLUE"");
//   printf("alhpa.real = %lf, alhpa.imag = %lf\n", alpha.real(), alpha.imag());
//   gpu_vector_norm2(temp_vec1, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of temp_vec1 is %lf\n", norm);
//   gpu_vector_norm2(temp_vec4, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of temp_vec4 is %lf\n", norm);
//   printf(CLR"");
// #endif


  if (if_odd_converge(iter_x_odd, temp_vec5, target_b, temp_vec2, temp_vec3, temp_vec4, gauge, d_kappa, d_coeff, d_norm1, d_norm2, param, kappa)) { // donnot use temp_vec1 !!!!
    return true;
  }

  // <r, r> is in numerator
  mpi_comm->interprocess_inner_prod_barrier(temp_vec1, temp_vec1, d_denominator, half_vol);  // <r', r'>

  checkCudaErrors(cudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost));
  beta = denominator / numerator;
  checkCudaErrors(cudaMemcpy(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice));
  // p = r' + \beta p
  gpu_sclar_multiply_vector (p_vec, d_beta, half_vol); // p_vec = \beta p_vec
  one = Complex(1, 0);
  checkCudaErrors(cudaMemcpy(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, p_vec, d_coeff, half_vol); // p <-- r' + \beta p

  checkCudaErrors(cudaMemcpy(resid_vec, temp_vec1, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice));  // r <--- r'

  return false;
}

// cg_odd
bool odd_cg_inverter (void* iter_x_odd, void* target_b, void* resid_vec, void* p_vec, \
  void* temp_vec1, void* temp_vec2, void* temp_vec3, void* temp_vec4, void* temp_vec5, \
  void* gauge, QcuParam *param, double kappa, void* d_kappa, \
  void* d_alpha, void* d_beta, void* d_denominator, void* d_numerator, void* d_coeff, void* d_norm1, void* d_norm2
) {
  // residential vector      resid_vec
  // p_vector

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;
  bool converge = false;

  Complex h_coeff;
  // x <---- 0
  clear_vector (iter_x_odd, half_vol * Ns * Nc);
  // b - Ax --->r
  // first : b----> r
  checkCudaErrors(cudaMemcpy(resid_vec, target_b, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice));
  // second: Ax ---> temp_vec4
  full_odd_solver_dslash (temp_vec4, iter_x_odd, temp_vec1, temp_vec2, temp_vec3, gauge, d_kappa, param, kappa);
  h_coeff = Complex(-1, 0);
  checkCudaErrors(cudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));
  // last: r <--- b-Ax
  mpi_comm->interprocess_saxpy_barrier(temp_vec4, resid_vec, d_coeff, half_vol);

// #ifdef DEBUG
//   printf(L_RED"");
//   double norm1;
//   gpu_vector_norm2(resid_vec, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of residential is %lf\n", norm1);

//   gpu_vector_norm2(target_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of new b is %lf\n", norm1);
//   printf(CLR"");
// #endif

  // If converge return x
  if (if_odd_converge(iter_x_odd, temp_vec4, target_b, temp_vec1, temp_vec2, temp_vec3, gauge, d_kappa, d_coeff, d_norm1, d_norm2, param, kappa)) {
    printf("cg suceess!\n");
    goto odd_cg_end;
  }

#ifdef DEBUG
  printf(RED"first iteration passed\n");
  printf(CLR"");
#endif
  // p <----r
  checkCudaErrors(cudaMemcpy(p_vec, resid_vec, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice));


// #ifdef DEBUG
//   printf(L_RED"");
//   gpu_vector_norm2(p_vec, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of p_vec is %lf\n", norm1);
//   printf(CLR"");
// #endif

  for (int i = 0; i < half_vol; i++) {
#ifdef DEBUG
    printf(RED"iteration %d\n", i+1);
    printf(CLR"");
#endif

#ifdef DEBUG
    double norm1;
    printf(L_RED"");
    gpu_vector_norm2(iter_x_odd, temp_vec5, half_vol, d_norm1);
    checkCudaErrors(cudaMemcpy(&norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
    printf("norm2 of ----iter_x_odd is %lf\n", norm1);

    gpu_vector_norm2(target_b, temp_vec5, half_vol, d_norm1);
    checkCudaErrors(cudaMemcpy(&norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
    printf("norm2 of ----target_b is %lf\n", norm1);
    printf(CLR"");
#endif


    converge = odd_cg_iter(iter_x_odd, target_b, resid_vec, p_vec, 
      temp_vec1, temp_vec2, temp_vec3, temp_vec4, temp_vec5, 
      gauge, param, kappa, 
      d_kappa, d_alpha, d_beta, d_denominator, d_numerator, d_coeff, d_norm1, d_norm2);
    if (converge) {
      printf("odd cg success! %d iterations\n", i+1);
      break;
    }
  }


odd_cg_end:
  return converge;
}



// modify b, half-length vector
void generate_new_b_odd(void* new_b, void* origin_odd_b, void* origin_even_b, void* temp_vec1, void* temp_vec2, void* temp_vec3, void* gauge, void* d_kappa, void* d_coeff, QcuParam *param, double kappa = 1.0) {
  // be + bo ----> be + new bo
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  Complex h_kappa(kappa, 0);
  Complex h_coeff;


  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;
  int parity;
  int dagger_flag;

  // even b ----> temp_vec1
  checkCudaErrors(cudaMemcpy(temp_vec1, origin_even_b, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice));

// #ifdef DEBUG
//   double norm;
//   double* d_norm1;
//   cudaMalloc(&d_norm1, sizeof(double));
//   printf(RED"");
//   gpu_vector_norm2(origin_odd_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> origin_odd_b is %lf\n", norm);
// #endif

  parity = 0;
  invertCloverDslashHalf (temp_vec1, nullptr, gauge, param, parity); // A^{-1}_{ee}b_{e} ---> temp_vec1

// #ifdef DEBUG
//   // double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(temp_vec1, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> origin_odd_b is %lf\n", norm);

//   gpu_vector_norm2(temp_vec1, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> origin_odd_b is %lf\n", norm);
// #endif

  parity = 1;
  dagger_flag = 0;
  callWilsonDslashNaive (new_b, temp_vec1, gauge, param, parity, dagger_flag); //  D_{oe}A^{-1}_{ee}b_{e} ----> new_b

// #ifdef DEBUG
//   // double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(new_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> -----new_b is %lf, kappa is %lf\n", norm, kappa);


//   gpu_vector_norm2(new_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> -----new_b is %lf, kappa is %lf\n", norm, kappa);
// #endif


  // kappa D_{oe}A^{-1}_{ee}b_{e}
  h_kappa = Complex(kappa, 0);
  checkCudaErrors(cudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_sax_barrier (new_b, d_kappa, half_vol);
// #ifdef DEBUG
//   // double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(new_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> new_b is %lf\n", norm);
//   gpu_vector_norm2(origin_odd_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> origin_odd_b is %lf\n", norm);
// #endif
  h_coeff = Complex(1, 0);
  checkCudaErrors(cudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(origin_odd_b, new_b, d_coeff, half_vol);

// #ifdef DEBUG
//   // double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(new_b, temp_vec3, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of <> new_b is %lf\n", norm);
// #endif
}








void cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param) {
// void cg_inverter(void* x_vector, void* b_vector, void *gauge, QcuParam *param){
  double kappa = 1.0;//) {
  int total_vol = param->lattice_size[0] * param->lattice_size[1] * param->lattice_size[2] * param->lattice_size[3];
  int half_vol = total_vol >> 1;
  bool if_end = false;
  // ptrs
  void* origin_even_b = b_vector;
  void* origin_odd_b = static_cast<void*>(static_cast<Complex*>(b_vector) + half_vol * Ns * Nc);
  void* even_x = x_vector;
  void* odd_x = static_cast<void*>(static_cast<Complex*>(x_vector) + half_vol * Ns * Nc);

  // Complex coeff;


  // ptrs need to allocate memory
  void* temp_vec1;
  void* temp_vec2;
  void* temp_vec3;
  void* temp_vec4;
  void* temp_vec5;

  void* p_vec;
  void* resid_vec;

  void* d_coeff;
  void* d_kappa;
  void* d_alpha;
  void* d_beta;
  void* d_denominator;
  void* d_numerator;
  void* d_norm1;
  void* d_norm2;
  void* new_b;

  // memory allocation
  checkCudaErrors(cudaMalloc(&temp_vec1, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&temp_vec2, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&temp_vec3, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&temp_vec4, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&temp_vec5, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&p_vec, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&resid_vec, sizeof(Complex) * half_vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&d_coeff, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_kappa, sizeof(Complex)));

  checkCudaErrors(cudaMalloc(&d_alpha, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_beta, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_denominator, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_numerator, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_norm1, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_norm2, sizeof(Complex)));

  checkCudaErrors(cudaMalloc(&new_b, sizeof(Complex) * half_vol * Ns * Nc));

// #ifdef DEBUG
//   double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(origin_odd_b, temp_vec5, half_vol, d_norm1);
//   checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
//   printf("norm2 of origin_odd_b is %lf\n", norm);
//   printf(CLR"");
// #endif

  // clear x, x <---0   void* odd_x 
  // clear_vector(odd_x, half_vol * Ns * Nc);
  // odd new_b
  generate_new_b_odd(temp_vec3, origin_odd_b, origin_even_b, temp_vec1, temp_vec2, temp_vec4, gauge, d_kappa, d_coeff, param, kappa);


#ifdef DEBUG
  double norm;
  printf(BLUE"");
  gpu_vector_norm2(temp_vec3, temp_vec5, half_vol, d_norm1);
  checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
  printf("norm2 of temp_vec3 is %lf\n", norm);
  printf(CLR"");
#endif


  // odd dagger D new_b
  odd_solver_dslash_dagger (new_b, temp_vec3, temp_vec1, temp_vec2, gauge, d_kappa, param, kappa) ;

#ifdef DEBUG
  // double norm;
  printf(BLUE"");
  gpu_vector_norm2(new_b, temp_vec5, half_vol, d_norm1);
  checkCudaErrors(cudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost));
  printf("norm2 of new_b is %lf\n", norm);
  printf(CLR"");
#endif

#ifdef DEBUG
  printf(RED"new odd b generated\n");
  printf(CLR"");
#endif



  if_end = odd_cg_inverter (odd_x, new_b, resid_vec, p_vec, \
                            temp_vec1, temp_vec2, temp_vec3, temp_vec4, temp_vec5, \
                            gauge, param, kappa, d_kappa, d_alpha, d_beta, \
                            d_denominator, d_numerator, d_coeff, d_norm1, d_norm2
  );

  if (if_end) {
    goto cg_end;
  }




cg_end:
  checkCudaErrors(cudaFree(temp_vec1));
  checkCudaErrors(cudaFree(temp_vec2));
  checkCudaErrors(cudaFree(temp_vec3));
  checkCudaErrors(cudaFree(temp_vec4));
  checkCudaErrors(cudaFree(temp_vec5));
  checkCudaErrors(cudaFree(p_vec));
  checkCudaErrors(cudaFree(resid_vec));
  checkCudaErrors(cudaFree(d_coeff));
  checkCudaErrors(cudaFree(d_kappa));

  checkCudaErrors(cudaFree(d_alpha));
  checkCudaErrors(cudaFree(d_beta));
  checkCudaErrors(cudaFree(d_denominator));
  checkCudaErrors(cudaFree(d_numerator));
  checkCudaErrors(cudaFree(d_norm1));
  checkCudaErrors(cudaFree(d_norm2));

  checkCudaErrors(cudaFree(new_b));
}