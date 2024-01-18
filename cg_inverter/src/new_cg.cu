#include "qcu.h"
#include "qcu_clover_dslash.cuh"
#include "qcu_communicator.cuh"
#include "qcu_complex.cuh"
#include "qcu_complex_computation.cuh"
#include "qcu_macro.cuh"
#include "qcu_shift_storage_complex.cuh"
#include "qcu_wilson_dslash_neo.cuh"
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <mpi.h>
#define DEBUG
#define COALESCED_CG

#define IGNORE_CLOVER

extern MPICommunicator *mpi_comm;
extern void *qcu_gauge;
extern int process_rank;


class TimerLog {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;

public:
  TimerLog() = default;
  void setTimer() { start_ = std::chrono::high_resolution_clock::now(); }
  void getTime(const char* str, int line) {
    end_ = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_)
            .count();
    // printf("line #<%d>, %s", line, str);
    printf(
        "line<%d> %s time : %.9lf sec\n",
        line, str, double(duration) / 1e9);
  }
};

// function pointers
void (*wilsonDslashFunction)(void *fermion_out, void *fermion_in, void *gauge,
                             QcuParam *param, int parity, int dagger_flag);

void (*invertCloverDslashHalfFunction)(void *fermion_out, void *fermion_in,
                                       void *gauge, QcuParam *param,
                                       int parity);
void (*cloverVectorHalfFuntion)(void *fermion_out, void *fermion_in,
                                void *gauge, QcuParam *param, int parity);

__attribute__((constructor)) void init_function() {
#ifdef COALESCED_CG
  wilsonDslashFunction = callWilsonDslashCoalesce;
  invertCloverDslashHalfFunction = invertCloverDslashHalfCoalesced;
  cloverVectorHalfFuntion = cloverVectorHalfCoalesced;
#else
  wilsonDslashFunction = callWilsonDslashNaive;
  invertCloverDslashHalfFunction = invertCloverDslashHalf;
  cloverVectorHalfFuntion = cloverVectorHalf;
#endif
}

/**
 * @brief clear the Complex vector named vec of vector_length elements to zero
 * (kernel function)
 *
 * @param vec
 * @param vector_length
 * @return void
 */
static __global__ void clearVectorKernel(void *vec, int vector_length) {
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  int vol = blockDim.x * gridDim.x;
  Complex *src = static_cast<Complex *>(vec);

  if (thread_id >= vector_length) {
    return;
  }

  for (int i = thread_id; i < vector_length; i += vol) {
    src[i].clear2Zero();
  }
}

/**
 * @brief clear the Complex vector named vec of vector_length elements to zero
 * (host function)
 *
 * @param vec
 * @param vector_length
 * @return void
 */
static void clear_vector(void *vec, int vector_length) {
  int block_size = MAX_BLOCK_SIZE;
  int grid_size =
      (vector_length + block_size * Ns * Nc - 1) / (block_size * Ns * Nc);
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
void odd_matrix_mul_vector(void *output_Ax, void *input_x, void *temp_vec1,
                           void *temp_vec2, void *gauge, void *d_kappa,
                           QcuParam *param, int dagger_flag, double kappa) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  int parity; // when use Doe, parity is odd which means 1, else 0
  Complex h_coeff;

  // qcuCudaMemcpy(output_Ax, input_x, sizeof(Complex) * half_vol * Ns * Nc,
  checkCudaErrors(cudaMemcpyAsync(output_Ax, input_x,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice));

  // part1 begin
  parity = 1;
#ifndef IGNORE_CLOVER
  cloverVectorHalfFuntion(output_Ax, nullptr, gauge, param, parity);
#endif
  // part1 end

  // part2 begin
  parity = 0;
  wilsonDslashFunction(temp_vec1, input_x, gauge, param, parity, dagger_flag);
  parity = 0;
#ifndef IGNORE_CLOVER
  invertCloverDslashHalfFunction(temp_vec1, nullptr, gauge, param,
                                 parity); // clover invert
#endif
  parity = 1;
  wilsonDslashFunction(temp_vec2, temp_vec1, gauge, param, parity, dagger_flag);
  // part2 end

  h_coeff = Complex(-kappa * kappa, 0);
  // qcuCudaMemcpy(d_kappa, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_kappa, &h_coeff, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  // saxpy
  mpi_comm->interprocess_saxpy_barrier(temp_vec2, output_Ax, d_kappa,
                                       half_vol); // coeff temp2 + x --->x
}

void full_odd_matrix_mul_vector(void *output_Ax, void *input_x, void *temp_vec1,
                                void *temp_vec2, void *temp_vec3, void *gauge,
                                void *d_kappa, QcuParam *param, double kappa) {

  int dagger_flag;
  dagger_flag = 0;
  odd_matrix_mul_vector(temp_vec3, input_x, temp_vec1, temp_vec2, gauge,
                        d_kappa, param, dagger_flag, kappa);
  dagger_flag = 1;
  odd_matrix_mul_vector(output_Ax, temp_vec3, temp_vec1, temp_vec2, gauge,
                        d_kappa, param, dagger_flag, kappa);
}
// current_b is temporary
bool if_even_converge(void *current_x, void *current_b_buffer, void *target_b,
                      void *temp_vec1, void *temp_vec2, void *temp_vec3,
                      void *gauge, void *d_kappa, void *d_coeff, void *d_norm1,
                      void *d_norm2, QcuParam *param, double kappa) {

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

  // qcuCudaMemcpy(current_b_buffer, current_x,
  //               sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemcpyAsync(current_b_buffer, current_x,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice));
#ifndef IGNORE_CLOVER
  cloverVectorHalfFuntion(current_b_buffer, nullptr, gauge, param,
                          parity); // Ax ---> current_b_buffer
#endif

  // gpu_vector_norm2 (target_b, temp_vec3, half_vol, d_norm1);
  mpi_comm->interprocess_vector_norm(target_b, temp_vec3, half_vol, d_norm1);

  // qcuCudaMemcpy(temp_vec2, target_b, sizeof(Complex) * half_vol * Ns * Nc,
  checkCudaErrors(
      cudaMemcpyAsync(temp_vec2, target_b, sizeof(Complex) * half_vol * Ns * Nc,
                      cudaMemcpyDeviceToDevice)); // target_b -----> temp_vec2
  h_coeff = Complex(-1, 0);
  // qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_coeff, &h_coeff, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(
      current_b_buffer, temp_vec2, d_coeff,
      half_vol); // temp_vec2 <--- target_b - current_b

  // gpu_vector_norm2(temp_vec2, temp_vec3, half_vol, d_norm2);
  mpi_comm->interprocess_vector_norm(temp_vec2, temp_vec3, half_vol, d_norm2);

  // qcuCudaMemcpy(&h_norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost);
  // qcuCudaMemcpy(&h_norm1, d_norm1, sizeof(double), cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&h_norm2, d_norm2, sizeof(double),
                                  cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(&h_norm2, d_norm2, sizeof(double),
                                  cudaMemcpyDeviceToHost));
#ifdef DEBUG
  printf("rank = %d, even difference :norm = %g, h_norm2 = %g, h_norm1=%g\n",
         process_rank, h_norm2 / h_norm1, h_norm2, h_norm1);
#endif
  return (h_norm2 / h_norm1 < 7e-15); // which means converge
}

// new if converge function
bool if_odd_converge(void *current_r, void *target_b, void *temp_vec,
                     double h_b_norm, double *d_r_norm, int half_vol) {
  double h_r_norm;

  mpi_comm->interprocess_vector_norm(current_r, temp_vec, half_vol, d_r_norm);

  // qcuCudaMemcpy(&h_r_norm, d_r_norm, sizeof(double), cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&h_r_norm, d_r_norm, sizeof(double),
                                  cudaMemcpyDeviceToHost));

#ifdef DEBUG
  printf(
      "rank = %d, odd difference :norm = %g, \n\tr_norm = %g, \tb_norm = %g\n",
      process_rank, h_r_norm / h_b_norm, h_r_norm, h_b_norm);
#endif
  return (h_r_norm / h_b_norm < 1e-15); // which means converge
}

bool odd_cg_iter(void *iter_x_odd, void *target_b, void *resid_vec, void *p_vec,
                 void *temp_vec1, void *temp_vec2, void *temp_vec3,
                 void *temp_vec4, void *temp_vec5, void *gauge, QcuParam *param,
                 double kappa, void *d_kappa, void *d_alpha, void *d_beta,
                 void *d_denominator, void *d_numerator, void *d_coeff,
                 void *d_norm1, void *d_norm2, double h_norm_b, int round) {

  // timer
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

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
  Complex one(1, 0);

#ifdef DEBUG
  TimerLog timer_log;
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  // <r, r>--->denominator
  mpi_comm->interprocess_inner_prod_barrier(resid_vec, resid_vec, d_numerator,
                                            half_vol); // <r, r> --> d_numerator
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("inner prod", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  full_odd_matrix_mul_vector(temp_vec4, p_vec, temp_vec1, temp_vec2, temp_vec3,
                             gauge, d_kappa, param, kappa);
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("full MvV ", __LINE__);
  }
#endif
  // Ap --->temp_vec4

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  mpi_comm->interprocess_inner_prod_barrier(
      p_vec, temp_vec4, d_denominator, half_vol); // <p, Ap> --> d_denominator
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("inner prod ", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  // qcuCudaMemcpy(&numerator, d_numerator, sizeof(Complex),
  //               cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&numerator, d_numerator, sizeof(Complex),
                                  cudaMemcpyDeviceToHost));
  // qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex),
  //               cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&denominator, d_denominator, sizeof(Complex),
                                  cudaMemcpyDeviceToHost));

  alpha = numerator / denominator;
  // qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_alpha, &alpha, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("3 memcpy and alpha calc ", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  mpi_comm->interprocess_saxpy_barrier(p_vec, iter_x_odd, d_alpha,
                                       half_vol); // x = x + \alpha p
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("saxpy ", __LINE__);
  }
#endif
  // qcuCudaMemcpy(temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice); // copy r to temp_vec1  r'=r

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  checkCudaErrors(cudaMemcpyAsync(
      temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc,
      cudaMemcpyDeviceToDevice)); // copy r to temp_vec1  r'=r
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("memcpyAsync ", __LINE__);
  }
#endif

  alpha = alpha * Complex(-1, 0);
  // qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  checkCudaErrors(cudaMemcpyAsync(d_alpha, &alpha, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("memcpyAsync ", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  mpi_comm->interprocess_saxpy_barrier(
      temp_vec4, temp_vec1, d_alpha,
      half_vol); // temp_vec4 = Ap, r'=r'-\alpha Ap------>temp_vec1

#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("saxpy ", __LINE__);
  }
#endif
  // donnot use temp1, as it is used
  // if (if_odd_converge(resid_vec, target_b, temp_vec2, half_vol,
  // static_cast<double*>(d_norm1), static_cast<double*>(d_norm2))) {
  //   return true;
  // }

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  if (if_odd_converge(resid_vec, target_b, temp_vec2, h_norm_b,
                      static_cast<double *>(d_norm1), half_vol)) {
    return true;
  }
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("test if converge ", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  // <r, r> is in numerator
  mpi_comm->interprocess_inner_prod_barrier(temp_vec1, temp_vec1, d_denominator,
                                            half_vol); // <r', r'>
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("inner prod ", __LINE__);
  }
#endif
  // qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex),
  //               cudaMemcpyDeviceToHost);

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  checkCudaErrors(cudaMemcpyAsync(&denominator, d_denominator, sizeof(Complex),
                                  cudaMemcpyDeviceToHost));
  beta = denominator / numerator;
  // qcuCudaMemcpy(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(
      cudaMemcpyAsync(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice));
  // p = r' + \beta p
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("2 memcpyAsync ", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  gpu_sclar_multiply_vector(p_vec, d_beta, half_vol); // p_vec = \beta p_vec
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("sax ", __LINE__);
  }
#endif

  one = Complex(1, 0);

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  // qcuCudaMemcpy(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(
      cudaMemcpyAsync(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice));
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("memcpyAsync ", __LINE__);
  }
#endif


#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, p_vec, d_coeff,
                                       half_vol); // p <-- r' + \beta p
#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("saxpy ", __LINE__);
  }
#endif

#ifdef DEBUG
  if (round == 2) {
    timer_log.setTimer();
  }
#endif
  // qcuCudaMemcpy(resid_vec, temp_vec1, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice); // r <--- r'
  checkCudaErrors(cudaMemcpyAsync(resid_vec, temp_vec1,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice)); // r <--- r'

#ifdef DEBUG
  if (round == 2) {
    timer_log.getTime("memcpyAsync ", __LINE__);
  }
#endif
  return false;
}

bool even_cg_iter(void *iter_x_odd, void *target_b, void *resid_vec,
                  void *p_vec, void *temp_vec1, void *temp_vec2,
                  void *temp_vec3, void *temp_vec4, void *temp_vec5,
                  void *gauge, QcuParam *param, double kappa, void *d_kappa,
                  void *d_alpha, void *d_beta, void *d_denominator,
                  void *d_numerator, void *d_coeff, void *d_norm1,
                  void *d_norm2) {

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
  Complex one(1, 0);

  // <r, r>--->denominator
  mpi_comm->interprocess_inner_prod_barrier(resid_vec, resid_vec, d_numerator,
                                            half_vol); // <r, r> --> d_numerator

  // qcuCudaMemcpy(temp_vec4, p_vec, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemcpyAsync(temp_vec4, p_vec,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice));
#ifndef IGNORE_CLOVER
  cloverVectorHalfFuntion(temp_vec4, nullptr, gauge, param,
                          0); // Ap --->temp_vec4
#endif
  mpi_comm->interprocess_inner_prod_barrier(
      p_vec, temp_vec4, d_denominator, half_vol); // <p, Ap> --> d_denominator

  // qcuCudaMemcpy(&numerator, d_numerator, sizeof(Complex),
  //               cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&numerator, d_numerator, sizeof(Complex),
                                  cudaMemcpyDeviceToHost));
  // qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex),
  //               cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&denominator, d_denominator, sizeof(Complex),
                                  cudaMemcpyDeviceToHost));

  alpha = numerator / denominator;
  // qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_alpha, &alpha, sizeof(Complex),
                                  cudaMemcpyHostToDevice));

  mpi_comm->interprocess_saxpy_barrier(p_vec, iter_x_odd, d_alpha,
                                       half_vol); // x = x + \alpha p

  // qcuCudaMemcpy(temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice); // copy r to temp_vec1  r'=r
  checkCudaErrors(cudaMemcpyAsync(
      temp_vec1, resid_vec, sizeof(Complex) * half_vol * Ns * Nc,
      cudaMemcpyDeviceToDevice)); // copy r to temp_vec1  r'=r

  alpha = alpha * Complex(-1, 0);
  // qcuCudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_alpha, &alpha, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(
      temp_vec4, temp_vec1, d_alpha,
      half_vol); // temp_vec4 = Ap, r'=r-\alpha Ap------>temp_vec1

  if (if_even_converge(iter_x_odd, temp_vec5, target_b, temp_vec2, temp_vec3,
                       temp_vec4, gauge, d_kappa, d_coeff, d_norm1, d_norm2,
                       param, kappa)) { // donnot use temp_vec1 !!!!
    return true;
  }

  // <r, r> is in numerator
  mpi_comm->interprocess_inner_prod_barrier(temp_vec1, temp_vec1, d_denominator,
                                            half_vol); // <r', r'>

  // qcuCudaMemcpy(&denominator, d_denominator, sizeof(Complex),
  //               cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&denominator, d_denominator, sizeof(Complex),
                                  cudaMemcpyDeviceToHost));
  beta = denominator / numerator;
  // qcuCudaMemcpy(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(
      cudaMemcpyAsync(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice));
  // p = r' + \beta p
  gpu_sclar_multiply_vector(p_vec, d_beta, half_vol); // p_vec = \beta p_vec
  one = Complex(1, 0);
  // qcuCudaMemcpy(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(
      cudaMemcpyAsync(d_coeff, &one, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, p_vec, d_coeff,
                                       half_vol); // p <-- r' + \beta p

  // qcuCudaMemcpy(resid_vec, temp_vec1, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice); // r <--- r'
  checkCudaErrors(cudaMemcpyAsync(resid_vec, temp_vec1,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice)); // r <--- r'

  return false;
}

bool even_solver(void *iter_x_even, void *target_b, void *temp_vec,
                 QcuParam *param) {
  bool if_converge;
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;
  int parity;
  double difference;
  // qcuCudaMemcpy(iter_x_even, target_b, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemcpyAsync(iter_x_even, target_b,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice));

  parity = 0;
#ifndef IGNORE_CLOVER
  invertCloverDslashHalfFunction(iter_x_even, nullptr, nullptr, param, parity);
#endif
  return true;
}

// cg_odd
// TODO: add a param to for norm_b
bool odd_cg_inverter(void *iter_x_odd, void *target_b, void *resid_vec,
                     void *p_vec, void *temp_vec1, void *temp_vec2,
                     void *temp_vec3, void *temp_vec4, void *temp_vec5,
                     void *gauge, QcuParam *param, double kappa, void *d_kappa,
                     void *d_alpha, void *d_beta, void *d_denominator,
                     void *d_numerator, void *d_coeff, void *d_norm1,
                     void *d_norm2, double h_norm_b) {

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  bool converge = false;
  Complex h_coeff;

  clear_vector(iter_x_odd, half_vol * Ns * Nc); // x <-- 0
  // b - Ax --->r
  // qcuCudaMemcpy(resid_vec, target_b, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice); // r <-- b
  checkCudaErrors(cudaMemcpyAsync(resid_vec, target_b,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice)); // r <-- b
  // second: Ax ---> temp_vec4
  full_odd_matrix_mul_vector(temp_vec4, iter_x_odd, temp_vec1, temp_vec2,
                             temp_vec3, gauge, d_kappa, param, kappa);

  h_coeff = Complex(-1, 0);
  // qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_coeff, &h_coeff, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(temp_vec4, resid_vec, d_coeff,
                                       half_vol); // last: r <-- b-Ax

  // If converge return x
  converge = if_odd_converge(resid_vec, target_b, temp_vec1, h_norm_b,
                             static_cast<double *>(d_norm1), half_vol);
  if (converge) {
    printf("cg suceess!\n");
    return converge;
  }

#ifdef DEBUG
  printf(RED "first iteration passed\n");
  printf(CLR "");
#endif
  // p <-- r
  // qcuCudaMemcpy(p_vec, resid_vec, sizeof(Complex) * half_vol * Ns * Nc,
  //               cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemcpyAsync(p_vec, resid_vec,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice));

  for (int i = 0; i < half_vol; i++) {

#ifdef DEBUG
    printf(RED "iteration %d", i + 1);
    printf(CLR "");
    auto start = std::chrono::high_resolution_clock::now();
#endif

    converge = odd_cg_iter(
        iter_x_odd, target_b, resid_vec, p_vec, temp_vec1, temp_vec2, temp_vec3,
        temp_vec4, temp_vec5, gauge, param, kappa, d_kappa, d_alpha, d_beta,
        d_denominator, d_numerator, d_coeff, d_norm1, d_norm2, h_norm_b, i);

    if (converge) {
      printf("odd cg success! %d iterations\n", i + 1);
      break;
    }
#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("odd iteration 1 time total time : %.9lf sec\n",
           double(duration) / 1e9);
#endif
  }

  return converge;
}

void generate_new_b_even(void *new_even_b, void *origin_even_b, void *res_odd_x,
                         void *gauge, void *d_kappa, void *d_coeff,
                         QcuParam *param, double kappa) {

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
  // qcuCudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_kappa, &h_kappa, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  // D_{eo}x_{o} ----> new_even_b
  wilsonDslashFunction(new_even_b, res_odd_x, gauge, param, parity,
                       dagger_flag);
  // kappa D_{eo}x_{o} ----> new_even_b
  mpi_comm->interprocess_sax_barrier(new_even_b, d_kappa, half_vol);

  h_coeff = Complex(1, 0);
  // qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_coeff, &h_coeff, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  // kappa D_{eo}x_{o} + even_b ----> new_even_b
  mpi_comm->interprocess_saxpy_barrier(origin_even_b, new_even_b, d_coeff,
                                       half_vol);
}

// modify b, half-length vector
void generate_new_b_odd(void *new_b, void *origin_odd_b, void *origin_even_b,
                        void *temp_vec1, void *temp_vec2, void *temp_vec3,
                        void *gauge, void *d_kappa, void *d_coeff,
                        QcuParam *param, double kappa) {

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
  // qcuCudaMemcpy(temp_vec1, origin_even_b, sizeof(Complex) * half_vol * Ns *
  // Nc,
  //               cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemcpyAsync(temp_vec1, origin_even_b,
                                  sizeof(Complex) * half_vol * Ns * Nc,
                                  cudaMemcpyDeviceToDevice));

  parity = 0;
#ifndef IGNORE_CLOVER
  invertCloverDslashHalfFunction(temp_vec1, nullptr, gauge, param,
                                 parity); // A^{-1}_{ee}b_{e} ---> temp_vec1
#endif

  parity = 1;
  dagger_flag = 0;
  wilsonDslashFunction(new_b, temp_vec1, gauge, param, parity,
                       dagger_flag); //  D_{oe}A^{-1}_{ee}b_{e} ----> new_b

  // kappa D_{oe}A^{-1}_{ee}b_{e}
  h_kappa = Complex(kappa, 0);
  // qcuCudaMemcpy(d_kappa, &h_kappa, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_kappa, &h_kappa, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  mpi_comm->interprocess_sax_barrier(new_b, d_kappa, half_vol);

  h_coeff = Complex(1, 0);
  // qcuCudaMemcpy(d_coeff, &h_coeff, sizeof(Complex), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaMemcpyAsync(d_coeff, &h_coeff, sizeof(Complex),
                                  cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(origin_odd_b, new_b, d_coeff, half_vol);
}

void cg_inverter(void *b_vector, void *x_vector, void *gauge, QcuParam *param) {
  double kappa = 0.125;

  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int half_vol = vol >> 1;

  double h_norm_b = 0;
#ifdef DEBUG
  printf("begin func cg, begin .....\n");
#endif

#ifdef COALESCED_CG
  gauge = qcu_gauge;
  void *origin_x_vector = x_vector;
  void *coalesced_b_vector;
  void *coalesced_x_vector;

  qcuCudaMalloc(&coalesced_b_vector, sizeof(Complex) * vol * Ns * Nc);
  qcuCudaMalloc(&coalesced_x_vector, sizeof(Complex) * vol * Ns * Nc);
  void *origin_vector_eo = b_vector;
  void *coalesced_vector_eo = coalesced_b_vector;
  shiftVectorStorageTwoDouble(coalesced_vector_eo, origin_vector_eo,
                              TO_COALESCE, Lx, Ly, Lz, Lt);
  origin_vector_eo = static_cast<void *>(static_cast<Complex *>(b_vector) +
                                         half_vol * Ns * Nc);
  coalesced_vector_eo = static_cast<void *>(
      static_cast<Complex *>(coalesced_b_vector) + half_vol * Ns * Nc);
  ;
  shiftVectorStorageTwoDouble(coalesced_vector_eo, origin_vector_eo,
                              TO_COALESCE, Lx, Ly, Lz, Lt);

  x_vector = coalesced_x_vector;
  b_vector = coalesced_b_vector;
#endif

  bool if_end = false;
  // ptrs doesn't need new memory
  void *origin_even_b;
  void *origin_odd_b;
  void *even_x;
  void *odd_x;

  // ptrs need to allocate memory
  void *temp_vec1;
  void *temp_vec2;
  void *temp_vec3;
  void *temp_vec4;
  void *temp_vec5;

  void *p_vec;
  void *resid_vec;

  void *d_coeff;
  void *d_kappa;
  void *d_alpha;
  void *d_beta;
  void *d_denominator;
  void *d_numerator;
  void *d_norm1;
  void *d_norm2;
  void *new_b;
  void *d_norm_b;
  int dagger_flag;

  origin_even_b = b_vector;
  origin_odd_b = static_cast<void *>(static_cast<Complex *>(b_vector) +
                                     half_vol * Ns * Nc);
  even_x = x_vector;
  odd_x = static_cast<void *>(static_cast<Complex *>(x_vector) +
                              half_vol * Ns * Nc);

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
  qcuCudaMalloc(&d_norm_b, sizeof(double));
  // odd new_b
  generate_new_b_odd(temp_vec3, origin_odd_b, origin_even_b, temp_vec1,
                     temp_vec2, temp_vec4, gauge, d_kappa, d_coeff, param,
                     kappa);

  // odd dagger D new_b
  dagger_flag = 1;
  odd_matrix_mul_vector(new_b, temp_vec3, temp_vec1, temp_vec2, gauge, d_kappa,
                        param, dagger_flag, kappa);

  mpi_comm->interprocess_vector_norm(new_b, temp_vec1, half_vol, d_norm_b);
  // qcuCudaMemcpy(&h_norm_b, d_norm_b, sizeof(double), cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaMemcpyAsync(&h_norm_b, d_norm_b, sizeof(double),
                                  cudaMemcpyDeviceToHost));

  if_end = odd_cg_inverter(odd_x, new_b, resid_vec, p_vec, temp_vec1, temp_vec2,
                           temp_vec3, temp_vec4, temp_vec5, gauge, param, kappa,
                           d_kappa, d_alpha, d_beta, d_denominator, d_numerator,
                           d_coeff, d_norm1, d_norm2, h_norm_b);

  if (!if_end) {
    printf("odd cg failed, donnot do even cg anymore, then exit\n");
    goto cg_end;
  }

  // even b
  generate_new_b_even(new_b, origin_even_b, odd_x, gauge, d_kappa, d_coeff,
                      param, kappa);

  if_end = even_solver(even_x, new_b, temp_vec1, param);
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
  qcuCudaFree(d_norm_b);
#ifdef COALESCED_CG
  x_vector = origin_x_vector;

  origin_vector_eo = x_vector;
  coalesced_vector_eo = coalesced_x_vector;
  shiftVectorStorageTwoDouble(origin_vector_eo, coalesced_vector_eo,
                              TO_NON_COALESCE, Lx, Ly, Lz, Lt);
  origin_vector_eo = static_cast<void *>(static_cast<Complex *>(x_vector) +
                                         half_vol * Ns * Nc);
  coalesced_vector_eo = static_cast<void *>(
      static_cast<Complex *>(coalesced_x_vector) + half_vol * Ns * Nc);
  shiftVectorStorageTwoDouble(origin_vector_eo, coalesced_vector_eo,
                              TO_NON_COALESCE, Lx, Ly, Lz, Lt);
  //   shiftVectorStorageTwoDouble(fermion_out, coalesced_fermion_out,
  //   TO_NON_COALESCE, Lx, Ly, Lz, Lt);
  qcuCudaFree(coalesced_b_vector);
  qcuCudaFree(coalesced_x_vector);
#endif
}
