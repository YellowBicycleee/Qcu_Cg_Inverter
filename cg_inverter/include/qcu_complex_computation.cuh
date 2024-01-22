#pragma once
#include "qcu_complex.cuh"



/**
 * @brief this function acts as y = a*x + y，a is scalar
 *
 * @param x       input             device ptr
 * @param y       input && output   device ptr
 * @param scalar  input             const Complex &
 * @param vol     input, full vector length is vol * Ns * Nc
 */
void gpu_saxpy(void *x, void *y, const Complex &scalar, int vol);




// xy inner product --->result (by partial result), vol means Lx * Ly * Lz * Lt
/**
 * @brief function gpu_inner_product calculates inner prod of vector x and y
 *         PAY ATTENTION: this function is out of date!!!!
 * @param x     complex vector, on device
 * @param y     complex vector, on device
 * @param result, output address, on device
 * @param partial_result 
 * @param vol       Lx*Ly*Lz*Lt
 */
void gpu_inner_product(void *x, void *y, void *result, void *partial_result,
                       int vol); // partial_result: reduction space

/**
 * @brief function gpu_inner_product_new calculates inner prod of vector x and y
 *         now, use this function instead of gpu_inner_product 
 * @param x       complex vector, on device, input parameter 
 * @param y       complex vector, on device, input parameter 
 * @param result ,   store result into this address, (on device), output parameter
 * @param partial_result 
 * @param vol: Lx * Ly * Lz * Lt
 */
void gpu_inner_product_new(void *x, void *y, void *result, void *partial_result,
                           int vol); // partial_result: reduction space

/**
 * @brief gpu_sclar_multiply_vector: x <- scalar*x
 * 
 * @param x 
 * @param scalar 
 * @param vol 
 */
void gpu_sclar_multiply_vector(void *x, const Complex& scalar, int vol);

/**
 * @brief gpu_vector_norm2 calc norm2 of vector
 * 
 * @param vector:         the vector which u want to calculate its norm2
 * @param temp_res        temp vector, use this to reduce
 * @param vector_length   the length of vector
 * @param result          result address
 */
void gpu_vector_norm2(void *vector, void *temp_res, int vector_length,
                      void *result);



// linear algebra
// functions version 2
// datd type complex<double>
// new gpu_saxpy
// void gpu_saxpy_todo(void *y, void *x, const Complex &scalar,
                    // std::size_t vector_length, cudaStream_t cuda_stream = NULL) {}