#pragma once

#define BEGIN_NAMESPACE(_) namespace _ {
#define END_NAMESPACE(_) }

enum DIMS { X_DIM = 0, Y_DIM, Z_DIM, T_DIM, Nd };
enum DIRS { FWD = 0, BWD, DIRECTIONS };

constexpr int Nc = 3;
constexpr int Ns = 4;

BEGIN_NAMESPACE(qcu)
struct QcuDesc {
  int lattice_size[4];
  int grid_size[4];
  QcuDesc(int Lx, int Ly, int Lz, int Lt, int Nx, int Ny, int Nz, int Nt) {
    lattice_size[0] = Lx;
    lattice_size[1] = Ly;
    lattice_size[2] = Lz;
    lattice_size[3] = Lt;

    grid_size[0] = Nx;
    grid_size[1] = Ny;
    grid_size[2] = Nz;
    grid_size[3] = Nt;
  }
};

END_NAMESPACE(qcu)