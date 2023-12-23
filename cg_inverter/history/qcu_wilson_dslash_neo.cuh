#pragma once

#include "qcu_dslash.cuh"
class WilsonDslash : public Dslash {
public:
  WilsonDslash(DslashParam& param) : Dslash(param){}
  virtual void calculateDslash(int dagger_flag = 0);
  virtual void calculateDslashNaive(int dagger_flag = 0);
  virtual void calculateDslashFull(double kappa, int dagger_flag = 0);
  virtual void dslashNaiveBenchmark(int dagger_flag = 0);
};

void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

void callWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

void callWilsonDslashCoalesce(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

void callWilsonDslashFull(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

// Benchmark
void callWilsonDslashBenchmark (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);
