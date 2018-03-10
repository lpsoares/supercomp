// Exemplo de programa para enviar e receber mensagem em MPI
// Coding style: Luciano Soares

#include <stdio.h>
#include <string.h>
#include <mpi.h>
int main(int argc, char ** argv) {
   int rank;
   char data[100];
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == 0) {
      strcpy(data,"Ola MPI");
      MPI_Send(data, 100, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
   }
   else if (rank == 1) {
      MPI_Recv(data, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("mensagem: %s\n",data);
   }
   MPI_Finalize();
   return 0;
}