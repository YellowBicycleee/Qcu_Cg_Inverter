#pragma once

#include "qcu_dslash.cuh"

class CloverDslash: public Dslash {
public:
  CloverDslash(DslashParam& param);
  void inverseCloverResult(void* p_fermion_out, void* p_invert_matrix,\
                      int Lx, int Ly, int Lz, int Lt, int parity); // real Lx Ly Lz Lt
  virtual void calculateDslash(int dagger_flag = 0);
  void cloverResult(void* p_fermion_out, void* p_clover_matrix,\
                    int Lx, int Ly, int Lz, int Lt, int parity);   // real Lx Ly Lz Lt
  void cloverResultCoalesced(void* p_fermion_out, void* p_clover_matrix, \
                    int Lx, int Ly, int Lz, int Lt, int parity);   // real Lx Ly Lz Lt
};

void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

void fullCloverDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag);


void cloverDslashOneRound(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag);


void invertCloverDslash (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param);

void newFullCloverDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag);

void invertCloverDslashHalf (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);//, int dagger_flag);
void cloverVectorHalf (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);



// void shiftCloverStorage(void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt);
void shiftCloverStorage(void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt);
void cloverVectorHalfCoalesced (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
void invertCloverDslashHalfCoalesced (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);

void callCloverDslashCoalesced(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

void callCloverDslashCoalesced_full(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);