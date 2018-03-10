#include "mpi.h"
#include <stdio.h>
#include <unistd.h>

int main (int argc, char *argv[]) {
   int size, rank;
   char messager[100], messages[100];
   int pair;
   MPI_Request request[2];
   MPI_Status status[2];
   
   MPI_Init(&argc, &argv); //inicia o MPI
   MPI_Comm_size(MPI_COMM_WORLD,&size); // recupera número de processos
   MPI_Comm_rank(MPI_COMM_WORLD,&rank); // recupera o indice (rank) do processo

   if (size%2) { // checa se o número de tarefas é par
      printf("Escolha um número par de processos\n");
      return(1);
   }

   printf("Processo %d/%d ativado\n", rank, size);
   sleep(1); // aguarda um pouco para mensagem ser vista na ordem

   pair = rank+1; // enviar para próximo processo
   if(pair==size) pair=0; // o último fala com o primeiro

   sprintf(messages,"Ola de processo %d",rank); // monta mensagem

   MPI_Irecv(&messager, 100, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
   MPI_Isend(&messages, 100, MPI_CHAR, pair, 0, MPI_COMM_WORLD, &request[1]);

   MPI_Waitall(2, request, status);  // bloqueie até todas as comunicações terminarem

   printf("Processo %d, recebeu de %d, a mensagem \"%s\"\n",rank,status[0].MPI_SOURCE,messager);

   MPI_Finalize();

}