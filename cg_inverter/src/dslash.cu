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
#include "qcu_complex_computation.cuh"
#include "qcu_point.cuh"
#include "qcu_communicator.cuh"
#include "qcu_clover_dslash.cuh"
#include "qcu_wilson_dslash_neo.cuh"
#include "qcu_wilson_dslash.cuh"

#define qcuPrint() { \
    printf("function %s line %d...\n", __FUNCTION__, __LINE__); \
}




void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
  // parity ---- invert_flag

  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // fullCloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // wilsonDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);
  callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);
  // callWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, 0);
}
void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag) {
  fullCloverDslashOneRound (fermion_out, fermion_in, gauge, param, dagger_flag);
}
