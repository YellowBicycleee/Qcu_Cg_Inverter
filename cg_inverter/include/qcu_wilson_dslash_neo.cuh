#pragma once

#include "qcu_dslash.cuh"
class WilsonDslash : public Dslash {
public:
  WilsonDslash(DslashParam& param) : Dslash(param){}
  virtual void calculateDslash(int invert_flag = 0);
  virtual void calculateDslashNaive(int invert_flag = 0);
  void calculateDoNothing(int invert_flag=0);
  void calculateDslashNaiveOnlyMemoryAccessing(int invert_flag = 0);
};

void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);

void callWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);
void callNop(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);
void calculateNaiveOnlyMemoryAccessing(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);