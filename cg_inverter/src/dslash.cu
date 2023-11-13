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
#include "qcu_shift_storage_complex.cuh"
#include "qcu_wilson_dslash_new_new.cuh"
#include <iostream>
using std::cout;
using std::endl;
#define qcuPrint() { \
    printf("function %s line %d...\n", __FUNCTION__, __LINE__); \
}


void* qcu_gauge;
void loadQcuGauge(void* gauge, QcuParam *param) {
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];

  checkCudaErrors(cudaMalloc(&qcu_gauge, sizeof(double) * Nd * Lx * Ly * Lz * Lt * (Nc-1) * Nc * 2));
  shiftGaugeStorageTwoDouble(qcu_gauge, gauge, TO_COALESCE, Lx, Ly, Lz, Lt);
}






void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
  // getDeviceInfo();
  // parity ---- invert_flag

  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // fullCloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // wilsonDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);

  // callWilsonDslash(fermion_out, fermion_in, qcu_gauge, param, parity, 0);
  // callWilsonDslashFull(fermion_out, fermion_in, gauge, param, parity, 0);


  // callWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, 0);
  // callNop(fermion_out, fermion_in, gauge, param, parity, 0);
  // calculateNaiveOnlyMemoryAccessing(fermion_out, fermion_in, gauge, param, parity, 0);
  // callNewDslash(fermion_out, fermion_in, gauge, param, parity, 0);
  callNewDslashCoalesced(fermion_out, fermion_in, gauge, param, parity, 0);
}
void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag) {
  fullCloverDslashOneRound (fermion_out, fermion_in, gauge, param, dagger_flag);
}
