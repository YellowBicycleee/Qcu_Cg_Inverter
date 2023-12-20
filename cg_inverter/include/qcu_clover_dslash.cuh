#pragma once

#include "qcu_dslash.cuh"

class CloverDslash: public Dslash {
public:
  CloverDslash(DslashParam& param);
  void inverseCloverResult(void* p_fermion_out, void* p_invert_matrix,\
                      int Lx, int Ly, int Lz, int Lt, int parity); // real Lx Ly Lz Lt
  virtual void calculateDslash(int invert_flag = 0);
  void cloverResult(void* p_fermion_out, void* p_clover_matrix,\
                    int Lx, int Ly, int Lz, int Lt, int parity);   // real Lx Ly Lz Lt
  void cloverResultCoalesced(void* p_fermion_out, void* p_clover_matrix, \
                    int Lx, int Ly, int Lz, int Lt, int parity);   // real Lx Ly Lz Lt
};

void callCloverDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);

void fullCloverDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag);


void cloverDslashOneRound(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag);

// fermion_in and fermion_out are the zero address of total vector
// void MmV_one_round (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param);
void MmV_one_round (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, void* temp);
void invertCloverDslash (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param);//, int invert_flag);
// void preCloverDslash (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag);
void newFullCloverDslashOneRound (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int invert_flag);

void invertCloverDslashHalf (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);//, int dagger_flag);
void cloverVectorHalf (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);



// void shiftCloverStorage(void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt);
void shiftCloverStorage(void* dst_vec, void* src_vec, int Lx, int Ly, int Lz, int Lt);
void cloverVectorHalfCoalesced (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);
void invertCloverDslashHalfCoalesced (void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity);

void callCloverDslashCoalesced_full(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);
void callCloverDslashCoalesced(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int invert_flag);