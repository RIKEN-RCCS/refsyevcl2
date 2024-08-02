#!/bin/bash

#PJM -L "node=512"               # num. of nodes
#PJM -L "rscgrp=large"           # int:1-12, small:1-384, large:385-55296
#PJM -L "elapse=20:00:00"        # time limit
#PJM -g PROJECT                  # group of project
#PJM --mpi "max-proc-per-node=4" # max num. of MPI process per node
#PJM -s

export PLE_MPI_STD_EMPTYFILE=off 
export OMP_NUM_THREADS=12 
export LD_LIBRARY_PATH=../src:../EigenExa-2.12/lib:$LD_LIBRARY_PATH

mpiexec ./test_refsyevcl 14 5 3
mpiexec ./test_refsyevcl 14 10 3
mpiexec ./test_refsyevcl 15 5 3
mpiexec ./test_refsyevcl 15 10 3
mpiexec ./test_refsyevcl 16 5 3
mpiexec ./test_refsyevcl 16 10 3
mpiexec ./test_refsyevcl 17 5 3
mpiexec ./test_refsyevcl 17 10 3
mpiexec ./test_refsyevcl 18 5 3
mpiexec ./test_refsyevcl 18 10 3

