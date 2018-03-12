/***********************************************************************************
* ARQUIVO: mpi_mm.c
* DESCRIÇÃO:
*   MPI Matrix Multiply - C Version
*   Neste código, o processo mestre distribui uma operação de multiplicacao
*   de matrizes para os processos escravos de tamanho numtasks-1.
*   AUTOR: Luciano Soares. Adaptado de Blaise Barney (4/05). Adaptado de Ros Leibensperger, Cornell Theory
*   Centro. Convertido para MPI: George L. Gusciora, MHPCC (1/95)
*   ÚLTIMA REVISAO: 12/03/18
********************************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NRA 162                /* numero de linhas na matriz A */
#define NCA 30                 /* numero de colunas na matriz A */
#define NCB 14                 /* numero de colunas na matriz B */
#define MASTER 0               /* ID da primeira tarefa */
#define FROM_MASTER 1          /* Tipo de mensagem 1 */
#define FROM_WORKER 2          /* Tipo de mensagem 2 */

int main (int argc, char *argv[]) {
   int numtasks,               /* numero de tarefas na particao */
   	taskid,                  /* identificador da tarefa */
   	numworkers,              /* numero de processos escravos */
   	source,                  /* identificador da mensagem de origem */
   	dest,                    /* identificador da mensagem de destino  */
   	mtype,                   /* tipo de mensagem */
   	rows,                    /* linhas da matriz A enviadas para cada processo escravo */
   	averow, extra, offset,   /* usado para determinar linhas enviadas para cada processo escravo */
   	i, j, k, rc;             /* diversos */

   double a[NRA][NCA],         /* matriz A a ser multiplicada */
   	b[NCA][NCB],             /* matriz B a ser multiplicada */
   	c[NRA][NCB];             /* matriz resultante C */

   MPI_Status status;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   if (numtasks < 2 ) {
     printf("Necessario ao minimo dois processos MPI. Encerrando.\n");
     MPI_Abort(MPI_COMM_WORLD, rc);
     exit(1);
   }
   numworkers = numtasks-1; // nesse modelo um 


/**************************** processo mestre ***********************************/
   if (taskid == MASTER) {

      printf("mpi_mm iniciou com %d processos.\n",numtasks);
      printf("Criando matrizes...\n");
      for (i=0; i<NRA; i++)
         for (j=0; j<NCA; j++)
            a[i][j]= (i+j)/10;
      for (i=0; i<NCA; i++)
         for (j=0; j<NCB; j++)
            b[i][j]= i*j*0.01;

      /* Envia dados da matriz para os processos escravos */
      averow = NRA/numworkers;
      extra  = NRA%numworkers;
      offset = 0;
      mtype  = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++) {

         rows = (dest <= extra) ? averow+1 : averow;   	
         printf("Enviando %d linhas para o processo %d, offset=%d\n",rows,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset][0], rows*NCA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&b, NCA*NCB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;

      }

      /* Recebendo resultados do processo trabalhador */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++) {

         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset][0], rows*NCB, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
         printf("Recebendo resultados do processo %d\n",source);

      }

      /* Exibe resultado */
      printf("******************************************************\n");
      printf("Matriz resultante:\n");
      for (i=0; i<NRA; i++) {
         printf("\n"); 
         for (j=0; j<NCB; j++) 
            printf("%5.1f ", c[i][j]);
      }
      printf("\n******************************************************\n");
   }


/***************************** processo escravo *********************************/
   if (taskid > MASTER) {

      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&a, rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&b, NCA*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

      for (k=0; k<NCB; k++)
         for (i=0; i<rows; i++) {
            c[i][k] = 0.0;
            for (j=0; j<NCA; j++)
               c[i][k] = c[i][k] + a[i][j] * b[j][k];
         }
      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&c, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }
   MPI_Finalize();
}

