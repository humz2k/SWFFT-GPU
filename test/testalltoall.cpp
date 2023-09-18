#include "alltoall.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(){
    MPI_Init(NULL,NULL);
    AllToAllGPU<CPUMPI,GPUPlanManager> alltoall(MPI_COMM_WORLD,8,64);
    MPI_Finalize();
    return 0;
}