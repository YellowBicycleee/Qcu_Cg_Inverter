#pragma once

#include "qcu_dslash.cuh"

class NewDslash : public Dslash {
public:
  NewDslash(DslashParam& param) : Dslash(param){}
  virtual void calculateDslash(int invert_flag = 0);
  virtual void calculateDslashCoalesced(int invert_flag = 0);
};

void callNewDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);
void callNewDslashCoalesced(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);