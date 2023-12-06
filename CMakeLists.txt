cmake_minimum_required(VERSION 3.8)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_BUILD_TYPE Release)
project(QCU LANGUAGES C CXX CUDA)

include_directories(/Users/cheng/myfile/git_file/qcu/Qcu_Cg_Inverter/cg_inverter/include)
#add_library(qcu SHARED src/dslash.cu)
#add_library(qcu SHARED src/dslash.cu  src/cg_inverter.cu src/qcu_complex_computation.cu )
#add_library(qcu SHARED  src/cg_inverter.cu src/dslash.cu src/qcu_clover_dslash.cu src/qcu_communicator.cu src/qcu_complex_computation.cu src/qcu_shift_storage_complex.cu src/qcu_wilson_dslash.cu src/qcu_wilson_dslash_neo.cu src/qcu_wilson_dslash_new_new.cu )
add_library(qcu SHARED  src/dslash.cu src/qcu_clover_dslash.cu src/qcu_communicator.cu src/qcu_complex_computation.cu src/qcu_shift_storage_complex.cu src/qcu_wilson_dslash.cu src/qcu_wilson_dslash_neo.cu src/qcu_wilson_dslash_new_new.cu src/new_cg.cu)

#target_compile_options(qcu PRIVATE "-Wall -Wextra")

find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)

