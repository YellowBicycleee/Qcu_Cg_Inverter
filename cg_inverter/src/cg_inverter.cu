#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "qcu.h"
#include <cuda_runtime.h>
#include "qcu_complex.cuh"
#include "qcu_complex_computation.cuh"


static void* d_right_hand_vec;
static void* d_matrix;

// CG inverter
void cg_inverter() {
  // test_computation();
}