#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>

int main(int argc, char** argv) {

  int rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int nums[8] = {5, 1, 2, 3, 7, 8, 4, 2};
  int result[2];

  MPI_Reduce(&nums[rank*2], &result, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Dados = [%d,%d,%d,%d,%d,%d,%d,%d]\n", nums[0],nums[1],nums[2],nums[3],nums[4],nums[5],nums[6],nums[7]);
    printf("Soma = [%d,%d]\n", result[0], result[1]);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}