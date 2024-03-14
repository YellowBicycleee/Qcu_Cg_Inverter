// #include <format>
#include "qcu_macro.cuh"
#include <assert.h>
// #include <iostream>
#include <cstdio>
#include <mpi.h>
// #define BEGIN_NAMESPACE(_) namespace _ {
// #define END_NAMESPACE(_) }

// enum DIMS { X_DIM = 0, Y_DIM, Z_DIM, T_DIM, Nd };
// enum DIRS { FWD = 0, BWD, DIRECTIONS };

enum CommOption { USE_MPI = 0, USE_NCCL, USE_GPU_AWARE_MPI };

#define MPI_CHECK(call)                                                        \
  do {                                                                         \
    int e = call;                                                              \
    if (e != MPI_SUCCESS) {                                                    \
      fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__);       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

BEGIN_NAMESPACE(qcu)

void Qcu_Init(int *argc, char ***argv) { MPI_CHECK(MPI_Init(argc, argv)); }

void Qcu_Finalize() { MPI_CHECK(MPI_Finalize()); }

void Qcu_Comm_rank(int *rank) {
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
}

void Qcu_Comm_size(int *size) {
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, size));
}

// void Qcu_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
//                MPI_Comm comm, MPI_Request *request) {
//   MPI_CHECK(MPI_Isend(buf, count, datatype, dest, tag, comm, request));
// }


struct MsgHandler
{
  // MPI_Request
  // MPI_Status
  // NCCL member
};


class QcuComm {
private:
  int processRank;                   // rank of current process
  int numProcess;                    // number of total processes
  int processCoord[Nd];              // coord of process in the grid
  int comm_grid_size[Nd];            // int[4]     = {Nx, Ny, Nz, Nt}
  int neighbor_rank[Nd][DIRECTIONS]; // int[4][2]


  int calcAdjProcess(int dim, int dir);

public:
  // force to use the constructor
  QcuComm(int Nx, int Ny, int Nz, int Nt);
  ~QcuComm() {}
};



END_NAMESPACE(qcu)