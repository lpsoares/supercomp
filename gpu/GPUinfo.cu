// Exemplo para o curso de Super Computacao
// Criado por: Luciano P. Soares (10 de Abril de 2018)

#include <stdio.h>
#include <stdlib.h>

//#include <cuda.h>
//#include <cuda_runtime.h>

/* Informacoes da GPU */
int main() {

   int dev_count;
   cudaGetDeviceCount(&dev_count);
   printf("Numero de devices (GPU) = %d\n\n", dev_count );

   cudaDeviceProp dev_prop;
   for (int i = 0; i < dev_count; i++) {
      printf("\tDevice (%d)\n", i);
      
      cudaGetDeviceProperties(&dev_prop, i);
      printf("\t\tNumero maximo de Bloco\n");
      printf("\t\t\t Dimensao maxima em x = %d, y = %d, z = %d\n", dev_prop.maxGridSize[0],dev_prop.maxGridSize[1],dev_prop.maxGridSize[2] );
      printf("\t\tNumero maximo de Threads por Bloco = %d\n", dev_prop.maxThreadsPerBlock );
      printf("\t\t\t Dimensao maxima em x = %d, y = %d, z = %d\n", dev_prop.maxThreadsDim[0],dev_prop.maxThreadsDim[1],dev_prop.maxThreadsDim[2] );
      printf("\t\tNumero maximo de Streaming Multiprocessors (SMs) = %d\n", dev_prop.multiProcessorCount );
      printf("\t\tFrequencia de Clock = %d\n", dev_prop.clockRate );
      printf("\t\tTamanho do Warp = %d\n", dev_prop.warpSize );
   
   }



   return 0;
}
