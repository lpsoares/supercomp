// Exemplo para o curso de Super Computacao
// Gera efeito de Bluer em imagens
// Criado por: Luciano P. Soares (10 de Abril de 2018)

#include <stdio.h>
#include <stdlib.h>
#include "image.h" 

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

/* Rotina para fazer a convolucao de um kernel de Blur */ 
__global__ void blur(int *input, int *output, int height, int width) {
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   int j=blockIdx.y*blockDim.y+threadIdx.y;
   if( i<height && j<width ) {   // Importante checar valor do i pois pode acessar fora do tamanho do vetor

      int total[3] = {0,0,0};
      int points = 0;

      for(int di = MAX(0, i - 1); di <= MIN(i + 1, height - 1); di++) {
         for(int dj = MAX(0, j - 1); dj <= MIN(j + 1, width - 1); dj++) {
            int pos = di*width + dj;
            total[0] += input[ pos*3   ];
            total[1] += input[ pos*3+1 ];
            total[2] += input[ pos*3+2 ];
            points++;
         }
      }

      output[(j*width + i)*3+0] = total[0]/points;
      output[(j*width + i)*3+1] = total[1]/points;
      output[(j*width + i)*3+2] = total[2]/points;
   }
}

/* Programa realiza Blur em uma imagem e salva em uma nova usando recursos de GPU */
int main(int argc, char** argv) {

   int *d_imageInput, *d_imageOutput;

   cudaError_t error;

   // Estruturas que organizam as imagens PGM
   PGMData *imagemIn  = (PGMData *)malloc(sizeof(PGMData));
   PGMData *imagemOut = (PGMData *)malloc(sizeof(PGMData));

   // Carrega Imagem RGB
   readImage(argv[1],imagemIn);

   // Cria imagem vazia
   createImage(imagemOut, imagemIn->row, imagemIn->col, imagemIn->max_intensity, 3);

   // Aloca vetores na memoria da GPU
   error = cudaMalloc((void **)&d_imageInput,imagemIn->row*imagemIn->col*imagemIn->channels*sizeof(int));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMalloc((void **)&d_imageOutput,imagemIn->row*imagemIn->col*imagemIn->channels*sizeof(int));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   // Copia valores da CPU para a GPU
   error = cudaMemcpy(d_imageInput, imagemIn->matrix, imagemIn->row*imagemIn->col*imagemIn->channels*sizeof(int), cudaMemcpyHostToDevice);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   // Libera memoria da CPU
   free(imagemIn->matrix);

   // Dimensoes para organizar na GPU
   dim3 dimGrid(ceil(imagemIn->row/(float)16.0), ceil(imagemIn->col/(float)16.0), 1);
   dim3 dimBlock(16, 16, 1);

   // Realiza conversao na GPU
   blur<<<dimGrid,dimBlock>>>(d_imageInput,d_imageOutput,imagemIn->row,imagemIn->col);

   // Retorna valores da memoria da GPU para a CPU
   error = cudaMemcpy(imagemOut->matrix, d_imageOutput, imagemIn->row*imagemIn->col*imagemIn->channels*sizeof(int), cudaMemcpyDeviceToHost);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   // Libera memoria da GPU
   cudaFree(d_imageInput);
   cudaFree(d_imageOutput);
   
   // Grava imagem convertida
   writeImage(argv[2],imagemOut,0);

   // Libera memoria da CPU
   free(imagemIn);
   free(imagemOut);

   return 0;
}
