// Exemplo de programa de Broadcast
// Coding style: Luciano Soares

#include <mpi.h>
#include <stdio.h>
#include <string.h>

void linear_bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {
   int rank;
   int size;

   MPI_Comm_rank(communicator, &rank);
   MPI_Comm_size(communicator, &size);

   if (rank==root) {
      int i;
      for (i = 0; i < size; i++) {
         if (i!=rank) MPI_Send(data, count, datatype, i, 0, communicator);
      }
   } else {
      MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
   }
}

int main(int argc, char ** argv) {
   int rank;
   char data[100];
   char processor[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   double starttime, endtime;
   double mpi_broadcast, meu_broadcast;
   int i;

   strcpy(data,"Ola do Master");

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Get_processor_name(processor, &name_len); // recuper nome do processor (host) e seu tamanho

   MPI_Barrier(MPI_COMM_WORLD);

   // BROADCAST MPI
   starttime = MPI_Wtime();
   for(i=0;i<10000;i++) {
      if (rank == 0) {
         MPI_Bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
      } else {
         MPI_Bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
      }
   }
   if (rank == 0) {
      mpi_broadcast = MPI_Wtime() - starttime;
      printf("Tempo decorrido de %f segundos com Broadcast do MPI\n",mpi_broadcast);
   }
   
   MPI_Barrier(MPI_COMM_WORLD);

   // BROADCAST MEU
   starttime = MPI_Wtime();
   for(i=0;i<10000;i++) {
      if (rank == 0) {
         linear_bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
      } else {
         linear_bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
      }
   }   
   if (rank == 0) {
      meu_broadcast = MPI_Wtime() - starttime;
      printf("Tempo decorrido de %f segundos com meu Broadcast\n",meu_broadcast);
      printf("\nBroadcast do MPI eh %f vezes mais rapido\n",meu_broadcast/mpi_broadcast);
   }

   endtime = MPI_Wtime();
   MPI_Finalize();
   return 0;
}
