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

// #define DEBUG


extern MPICommunicator *mpi_comm;

/**
 * @brief clear the Complex vector named vec of vector_length elements to zero (kernel function)
 * 
 * @param vec 
 * @param vector_length 
 * @return void
 */
static __global__ void clearVectorKernel(void* vec, int vector_length) {
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  int vol = blockDim.x * gridDim.x;
  Complex* src = static_cast<Complex*>(vec);

  if (thread_id >= vector_length) {
    return;
  }

  for (int i = thread_id; i < vector_length; i += vol) {
    src[i].clear2Zero();
  }
}

/**
 * @brief clear the Complex vector named vec of vector_length elements to zero (host function)
 * 
 * @param vec 
 * @param vector_length 
 * @return void
 */
static void clear_vector (void* vec, int vector_length) {
  int block_size = MAX_BLOCK_SIZE;
  int grid_size = (vector_length + block_size * Ns * Nc - 1) / (block_size * Ns * Nc);
  clearVectorKernel<<<grid_size, block_size>>>(vec, vector_length);
  qcuCudaDeviceSynchronize();
}


/**
 * @brief use this function to do one time MmV, when what to calc odd x
 * 
 * @param output_Ax result which means matrix A multiply vector x
 * @param input_x input vector x
 * @param temp_vec1 temporary space
 * @param temp_vec2 temporary space
 * @param gauge gauge
 * @param d_kappa kappa device pointer
 * @param param size of Lx Ly Lz Lt
 * @param dagger_flag 0 means no dagger, 1 means dagger
 * @param kappa kappa (double)
 */
void odd_matrix_mul_vector (void* output_Ax, void* input_x, void* temp_vec1, void* temp_vec2, void* gauge, void* d_kappa, QcuParam *param, int dagger_flag = 0, double kappa = 1.0) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  int parity;     // when use Doe, parity is odd which means 1, else 0
  Complex h_coeff;

  qcuCudaMemcpy(output_Ax, input_x, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);

  // part1 begin
  parity = 1;
  cloverVectorHalf (output_Ax, nullptr, gauge, param, parity);
  // part1 end

  // part2 begin
  parity = 0;
  callWilsonDslashNaive(temp_vec1, input_x, gauge, param, parity, dagger_flag);
  parity = 0;
  invertCloverDslashHalf (temp_vec1, nullptr, gauge, param, parity);  // clover invert
  parity = 1;
  callWilsonDslashNaive(temp_vec2, temp_vec1, gauge, param, parity, dagger_flag);
  // part2 end

  h_coeff = Complex(-kappa * kappa, 0);
  qcuCudaMemcpy(d_kappa, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  // saxpy
  mpi_comm->interprocess_saxpy_barrier(temp_vec2, output_Ax, d_kappa, half_vol);  // coeff temp2 + x --->x
}

void full_odd_matrix_mul_vector (void* output_Ax, void* input_x, void* temp_vec1, void* temp_vec2, void* temp_vec3, void* gauge, void* d_kappa, QcuParam *param, double kappa = 1.0) {

  int dagger_flag;
  dagger_flag = 0;
  odd_matrix_mul_vector (temp_vec3, input_x, temp_vec1, temp_vec2, gauge, \
                          d_kappa, param, dagger_flag, kappa);
  dagger_flag = 1;
  odd_matrix_mul_vector (output_Ax, temp_vec3, temp_vec1, temp_vec2, gauge, \
                          d_kappa, param, dagger_flag, kappa);
}
// current_b is temporary
bool if_even_converge(void* current_x, void* current_b_buffer, void* target_b, \
                    void* temp_vec1, void* temp_vec2, void* temp_vec3, \
                    void* gauge, void* d_kappa, void* d_coeff, \
                    void* d_norm1, void* d_norm2, QcuParam *param, double kappa = 1.0
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  Complex h_coeff;
  double h_norm1; // norm(target_b)
  double h_norm2; // norm(target_b - current_b)

  int parity = 0;

  cloverVectorHalf (current_b_buffer, nullptr, gauge, param, parity);  // Ax ---> temp_vec1

  gpu_vector_norm2 (target_b, temp_vec3, half_vol, d_norm1);

  qcuCudaMemcpy (temp_vec2, target_b, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);     // target_b -----> temp_vec2
  h_coeff = Complex(-1, 0);
  qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(current_b_buffer, temp_vec2, d_coeff, \
                half_vol); // temp_vec2 <--- target_b - current_b

  gpu_vector_norm2(temp_vec2, temp_vec3, half_vol, d_norm2);
  qcuCudaMemcpy(&h_norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost);
  qcuCudaMemcpy(&h_norm2, d_norm2, sizeof(double), cudaMemcpyDeviceToHost);
  return (h_norm2 / h_norm1 < 1e-13); // which means converge
}

// current_b is temporary
bool if_odd_converge(void* current_x, void* current_b_buffer, void* target_b, \
                    void* temp_vec1, void* temp_vec2, void* temp_vec3, \
                    void* gauge, void* d_kappa, void* d_coeff, \
                    void* d_norm1, void* d_norm2, QcuParam *param, double kappa = 1.0
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  Complex h_coeff;
  double h_norm1; // norm(target_b)
  double h_norm2; // norm(target_b - current_b)

  full_odd_matrix_mul_vector (current_b_buffer, current_x, \
                temp_vec1, temp_vec2, temp_vec3, gauge, d_kappa, param, kappa);

  gpu_vector_norm2 (target_b, temp_vec3, half_vol, d_norm1);

  qcuCudaMemcpy (temp_vec2, target_b, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);     // target_b -----> temp_vec2
  h_coeff = Complex(-1, 0);
  qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(current_b_buffer, temp_vec2, d_coeff, \
                half_vol); // temp_vec2 <--- target_b - current_b

  gpu_vector_norm2(temp_vec2, temp_vec3, half_vol, d_norm2);
  qcuCudaMemcpy(&h_norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost);
  qcuCudaMemcpy(&h_norm2, d_norm2, sizeof(double), cudaMemcpyDeviceToHost);
#ifdef DEBUG
  printf("difference %.32lf, norm1 = %lf, norm2 = %lf\n", \
              h_norm2 / h_norm1, h_norm1, h_norm2);
#endif
  return (h_norm2 / h_norm1 < 1e-13); // which means converge
}

bool odd_cg_iter(void* iter_x_odd, void* target_b, void* resid_vec, void* p_vec, \
        void* temp_vec1, void* temp_vec2, void* temp_vec3, void* temp_vec4, void* temp_vec5, \
        void* gauge, QcuParam *param, double kappa, void* d_kappa, \
        void* d_alpha, void* d_beta, void* d_denominator, void* d_numerator, \
        void* d_coeff, void* d_norm1, void* d_norm2
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  Complex alpha;
  Complex beta;
  Complex denominator;
  Complex numerator;
  Complex one(1,0);

  // <r, r>--->denominator
  mpi_comm->interprocess_inner_prod_barrier(resid_vec, resid_vec, \
                                    d_numerator, half_vol);  // <r, r> --> d_numerator


  full_odd_matrix_mul_vector (temp_vec4, p_vec, temp_vec1, \
                    temp_vec2, temp_vec3, gauge, d_kappa, param, kappa);
                    // Ap --->temp_vec4


  mpi_comm->interprocess_inner_prod_barrier(p_vec, temp_vec4, d_denominator, \
                    half_vol);  // <p, Ap> --> d_denominator

  qcuCudaMemcpy(&numerator, d_numerator, sizeof(Complex), cudaMemcpyDeviceToHost);
  qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost);
// #ifdef DEBUG
//   printf(RED"");
//   printf("numerator %lf %lf\n", numerator.real(), numerator.imag());
//   printf("denominator %lf %lf\n", denominator.real(), denominator.imag());
//   printf(CLR"");
// #endif

  alpha = numerator / denominator;
  qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);

  mpi_comm->interprocess_saxpy_barrier(p_vec, iter_x_odd, d_alpha, half_vol); // x = x + \alpha p

  qcuCudaMemcpy(temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice); // copy r to temp_vec1  r'=r

  alpha = alpha * Complex(-1, 0);
  qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(temp_vec4, temp_vec1, d_alpha, half_vol); // temp_vec4 = Ap, r'=r'-\alpha Ap------>temp_vec1

  if (if_odd_converge(iter_x_odd, temp_vec5, \
                      target_b, temp_vec2, temp_vec3,\
                      temp_vec4, gauge, d_kappa, d_coeff, \
                      d_norm1, d_norm2, param, kappa)
  ) { // donnot use temp_vec1 !!!!
    return true;
  }

  // <r, r> is in numerator
  mpi_comm->interprocess_inner_prod_barrier(temp_vec1, temp_vec1, \
                                            d_denominator, half_vol);  // <r', r'>

  qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost);
  beta = denominator / numerator;
  qcuCudaMemcpy(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice);
  // p = r' + \beta p
  gpu_sclar_multiply_vector (p_vec, d_beta, half_vol); // p_vec = \beta p_vec
  one = Complex(1, 0);
  qcuCudaMemcpy(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, p_vec, d_coeff, half_vol); // p <-- r' + \beta p

  qcuCudaMemcpy(resid_vec, temp_vec1, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);  // r <--- r'

  return false;
}




bool even_cg_iter(void* iter_x_odd, void* target_b, void* resid_vec, void* p_vec, \
        void* temp_vec1, void* temp_vec2, void* temp_vec3, void* temp_vec4, void* temp_vec5, \
        void* gauge, QcuParam *param, double kappa, void* d_kappa, \
        void* d_alpha, void* d_beta, void* d_denominator, void* d_numerator, \
        void* d_coeff, void* d_norm1, void* d_norm2
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  Complex alpha;
  Complex beta;
  Complex denominator;
  Complex numerator;
  Complex one(1,0);

  // <r, r>--->denominator
  mpi_comm->interprocess_inner_prod_barrier(resid_vec, resid_vec, \
                                    d_numerator, half_vol);  // <r, r> --> d_numerator


  qcuCudaMemcpy(temp_vec4, p_vec, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);
  cloverVectorHalf (temp_vec4, nullptr, gauge, param, 0);  // Ap --->temp_vec4


  mpi_comm->interprocess_inner_prod_barrier(p_vec, temp_vec4, d_denominator, \
                    half_vol);  // <p, Ap> --> d_denominator

  qcuCudaMemcpy(&numerator, d_numerator, sizeof(Complex), cudaMemcpyDeviceToHost);
  qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost);

  alpha = numerator / denominator;
  qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);

  mpi_comm->interprocess_saxpy_barrier(p_vec, iter_x_odd, d_alpha, half_vol); // x = x + \alpha p

  qcuCudaMemcpy(temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice); // copy r to temp_vec1  r'=r

  alpha = alpha * Complex(-1, 0);
  qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(temp_vec4, temp_vec1, d_alpha, half_vol); // temp_vec4 = Ap, r'=r-\alpha Ap------>temp_vec1

  if (if_even_converge(iter_x_odd, temp_vec5, \
                      target_b, temp_vec2, temp_vec3,\
                      temp_vec4, gauge, d_kappa, d_coeff, \
                      d_norm1, d_norm2, param, kappa)
  ) { // donnot use temp_vec1 !!!!
    return true;
  }

  // <r, r> is in numerator
  mpi_comm->interprocess_inner_prod_barrier(temp_vec1, temp_vec1, \
                                            d_denominator, half_vol);  // <r', r'>

  qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost);
  beta = denominator / numerator;
  qcuCudaMemcpy(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice);
  // p = r' + \beta p
  gpu_sclar_multiply_vector (p_vec, d_beta, half_vol); // p_vec = \beta p_vec
  one = Complex(1, 0);
  qcuCudaMemcpy(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, p_vec, d_coeff, half_vol); // p <-- r' + \beta p

  qcuCudaMemcpy(resid_vec, temp_vec1, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);  // r <--- r'

  return false;
}



// cg_even
bool even_cg_inverter (void* iter_x_even, void* target_b, void* resid_vec, void* p_vec,
  void* temp_vec1, void* temp_vec2, void* temp_vec3, void* temp_vec4, void* temp_vec5,\
  void* gauge, QcuParam *param, double kappa, void* d_kappa, \
  void* d_alpha, void* d_beta, void* d_denominator, void* d_numerator, void* d_coeff, void* d_norm1, void* d_norm2
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  int parity;
  bool if_converge = false;
  Complex h_coeff;

  clear_vector (iter_x_even, half_vol * Ns * Nc);  // x <-- 0
  // b - Ax --->r
  qcuCudaMemcpy (resid_vec, target_b, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);      // r <-- b

  qcuCudaMemcpy (temp_vec1, iter_x_even, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);  // x-->temp_vec1
  parity = 0;
  cloverVectorHalf (temp_vec1, nullptr, gauge, param, parity);  // Ax ---> temp_vec1

  if_converge =  if_even_converge(iter_x_even, temp_vec5, target_b, temp_vec1, \
                                  temp_vec2, temp_vec3, gauge, d_kappa, d_coeff, \
                                  d_norm1, d_norm2, param, kappa);
  if (if_converge) {
    return if_converge;
  }
  // then   r--->p
  qcuCudaMemcpy(p_vec, resid_vec, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);


  for (int i = 0; i < half_vol; i++) {
    if_converge = even_cg_iter(iter_x_even, target_b, resid_vec, p_vec, \
                          temp_vec1, temp_vec2, temp_vec3, temp_vec4, temp_vec5, \
                          gauge, param, kappa, d_kappa, d_alpha, d_beta, \
                          d_denominator, d_numerator, d_coeff, d_norm1, d_norm2);
    if (if_converge) {
      printf("even cg success! %d iterations\n", i+1);
      break;
    }
  }

  return if_converge;
}


// cg_odd
bool odd_cg_inverter (void* iter_x_odd, void* target_b, void* resid_vec, void* p_vec, \
  void* temp_vec1, void* temp_vec2, void* temp_vec3, void* temp_vec4, void* temp_vec5,\
  void* gauge, QcuParam *param, double kappa, void* d_kappa, \
  void* d_alpha, void* d_beta, void* d_denominator, void* d_numerator, void* d_coeff, void* d_norm1, void* d_norm2
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  bool converge = false;
  Complex h_coeff;

  clear_vector (iter_x_odd, half_vol * Ns * Nc);  // x <-- 0
  // b - Ax --->r
  qcuCudaMemcpy (resid_vec, target_b, sizeof(Complex) * half_vol * Ns * Nc, \
                cudaMemcpyDeviceToDevice);      // r <-- b
  // second: Ax ---> temp_vec4
  full_odd_matrix_mul_vector (temp_vec4, iter_x_odd, temp_vec1, \
                            temp_vec2, temp_vec3, gauge, d_kappa, param, kappa);

  h_coeff = Complex(-1, 0);
  qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(temp_vec4, resid_vec, d_coeff, \
                                        half_vol);  // last: r <-- b-Ax


  // If converge return x
  if (if_odd_converge(iter_x_odd, temp_vec4, target_b, temp_vec1, temp_vec2, temp_vec3, gauge, d_kappa, d_coeff, d_norm1, d_norm2, param, kappa)) {
    printf("cg suceess!\n");
    goto odd_cg_end;
  }

#ifdef DEBUG
  printf(RED"first iteration passed\n");
  printf(CLR"");
#endif
  // p <-- r
  qcuCudaMemcpy(p_vec, resid_vec, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice);


  for (int i = 0; i < half_vol; i++) {
#ifdef DEBUG
    printf(RED"iteration %d\n", i+1);
    printf(CLR"");
#endif


  // test inner prod
// #ifdef DEBUG
//   Complex temp;
//   mpi_comm->interprocess_inner_prod_barrier(target_b, target_b, temp_vec5, half_vol);
//   qcuCudaMemcpy(&temp, temp_vec5, sizeof(Complex), cudaMemcpyDeviceToHost);
//   printf(BLUE"temp.real = %lf, temp.imag = %lf\n", temp.real(), temp.imag());
//   printf(CLR"");
// #endif

    converge = odd_cg_iter(iter_x_odd, target_b, resid_vec, p_vec, \
                          temp_vec1, temp_vec2, temp_vec3, temp_vec4, temp_vec5, \
                          gauge, param, kappa, d_kappa, d_alpha, d_beta, \
                          d_denominator, d_numerator, d_coeff, d_norm1, d_norm2);

    if (converge) {
      printf("odd cg success! %d iterations\n", i+1);
      break;
    }
  }


odd_cg_end:
  return converge;
}

void generate_new_b_even (void* new_even_b, void* origin_even_b, void* res_odd_x, \
                          void* gauge, void* d_kappa, void* d_coeff, \
                        QcuParam *param, double kappa
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  int parity = 0;
  int dagger_flag = 0;
  Complex h_kappa(kappa, 0);
  Complex h_coeff;
  qcuCudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice);

  // D_{eo}x_{o} ----> new_even_b
  callWilsonDslashNaive (new_even_b, res_odd_x, gauge, param, parity, dagger_flag);
  // kappa D_{eo}x_{o} ----> new_even_b
  mpi_comm->interprocess_sax_barrier (new_even_b, d_kappa, half_vol);

  h_coeff = Complex(1, 0);
  qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  // kappa D_{eo}x_{o} + even_b ----> new_even_b
  mpi_comm->interprocess_saxpy_barrier(origin_even_b, new_even_b, d_coeff, half_vol);
}


// modify b, half-length vector
void generate_new_b_odd(void* new_b, void* origin_odd_b, void* origin_even_b, \
                        void* temp_vec1, void* temp_vec2, void* temp_vec3, \
                        void* gauge, void* d_kappa, void* d_coeff, \
                        QcuParam *param, double kappa = 1.0
) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  Complex h_kappa;
  Complex h_coeff;
  int parity;
  int dagger_flag;

  // even b ----> temp_vec1
  qcuCudaMemcpy(temp_vec1, origin_even_b, sizeof(Complex) * half_vol * Ns * Nc, cudaMemcpyDeviceToDevice);

  parity = 0;
  invertCloverDslashHalf (temp_vec1, nullptr, gauge, param, parity); // A^{-1}_{ee}b_{e} ---> temp_vec1

  parity = 1;
  dagger_flag = 0;
  callWilsonDslashNaive (new_b, temp_vec1, gauge, param, parity, dagger_flag); //  D_{oe}A^{-1}_{ee}b_{e} ----> new_b

  // kappa D_{oe}A^{-1}_{ee}b_{e}
  h_kappa = Complex(kappa, 0);
  qcuCudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_sax_barrier (new_b, d_kappa, half_vol);

  h_coeff = Complex(1, 0);
  qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  mpi_comm->interprocess_saxpy_barrier(origin_odd_b, new_b, d_coeff, half_vol);
}


void cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param) {
  double kappa = 1.0;

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  bool if_end = false;
  // ptrs doesn't need new memory
  void* origin_even_b;
  void* origin_odd_b;
  void* even_x;
  void* odd_x;

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

  int dagger_flag;

  origin_even_b = b_vector;
  origin_odd_b = static_cast<void*>(static_cast<Complex*>(b_vector) \
                  + half_vol * Ns * Nc);
  even_x = x_vector;
  odd_x = static_cast<void*>(static_cast<Complex*>(x_vector) + half_vol * Ns * Nc);

  // memory allocation
  qcuCudaMalloc(&temp_vec1, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&temp_vec2, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&temp_vec3, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&temp_vec4, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&temp_vec5, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&p_vec, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&resid_vec, sizeof(Complex) * half_vol * Ns * Nc);
  qcuCudaMalloc(&d_coeff, sizeof(Complex));
  qcuCudaMalloc(&d_kappa, sizeof(Complex));

  qcuCudaMalloc(&d_alpha, sizeof(Complex));
  qcuCudaMalloc(&d_beta, sizeof(Complex));
  qcuCudaMalloc(&d_denominator, sizeof(Complex));
  qcuCudaMalloc(&d_numerator, sizeof(Complex));
  qcuCudaMalloc(&d_norm1, sizeof(Complex));
  qcuCudaMalloc(&d_norm2, sizeof(Complex));

  qcuCudaMalloc(&new_b, sizeof(Complex) * half_vol * Ns * Nc);



  // clear x, x <---0   void* odd_x 
  // clear_vector(odd_x, half_vol * Ns * Nc);
  // odd new_b
  generate_new_b_odd(temp_vec3, origin_odd_b, origin_even_b, temp_vec1, temp_vec2, temp_vec4, gauge, d_kappa, d_coeff, param, kappa);


// #ifdef DEBUG
//   double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(temp_vec3, temp_vec5, half_vol, d_norm1);
//   qcuCudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost);
//   printf("norm2 of temp_vec3 is %lf\n", norm);
//   printf(CLR"");
// #endif


  // odd dagger D new_b
  dagger_flag = 1;
  odd_matrix_mul_vector (new_b, temp_vec3, temp_vec1, temp_vec2, \
                         gauge, d_kappa, param, dagger_flag, kappa);

// #ifdef DEBUG
//   // double norm;
//   printf(BLUE"");
//   gpu_vector_norm2(new_b, temp_vec5, half_vol, d_norm1);
//   qcuCudaMemcpy(&norm, d_norm1, sizeof(double), cudaMemcpyDeviceToHost);
//   printf("norm2 of new_b is %lf\n", norm);
//   printf(CLR"");
// #endif

#ifdef DEBUG
  printf(RED"new odd b generated\n");
  printf(CLR"");
#endif



  if_end = odd_cg_inverter (odd_x, new_b, resid_vec, p_vec, \
                            temp_vec1, temp_vec2, temp_vec3, temp_vec4, temp_vec5, \
                            gauge, param, kappa, d_kappa, d_alpha, d_beta, \
                            d_denominator, d_numerator, d_coeff, d_norm1, d_norm2
  );

  if (!if_end) {
    printf("odd cg failed, donnot do even cg anymore, then exit\n");
    goto cg_end;
  }

  // odd succeed

  // even b
  generate_new_b_even (new_b, origin_even_b, odd_x,
                            gauge, d_kappa, d_coeff, param, kappa);

  if_end = even_cg_inverter (even_x, new_b, resid_vec, p_vec, \
                            temp_vec1, temp_vec2, temp_vec3, temp_vec4, temp_vec5, \
                            gauge, param, kappa, d_kappa, d_alpha, d_beta, \
                            d_denominator, d_numerator, d_coeff, d_norm1, d_norm2
  );
  if (!if_end) {
    printf("even cg failed, then exit\n");
    goto cg_end;
  }
cg_end:
  qcuCudaFree(temp_vec1);
  qcuCudaFree(temp_vec2);
  qcuCudaFree(temp_vec3);
  qcuCudaFree(temp_vec4);
  qcuCudaFree(temp_vec5);
  qcuCudaFree(p_vec);
  qcuCudaFree(resid_vec);
  qcuCudaFree(d_coeff);
  qcuCudaFree(d_kappa);

  qcuCudaFree(d_alpha);
  qcuCudaFree(d_beta);
  qcuCudaFree(d_denominator);
  qcuCudaFree(d_numerator);
  qcuCudaFree(d_norm1);
  qcuCudaFree(d_norm2);

  qcuCudaFree(new_b);
}