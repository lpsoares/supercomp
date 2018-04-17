// Exemplo para o curso de Super Computacao
// Conversao de imagens coloridas para tons de cinza
// Criado por: Luciano P. Soares (10 de Abril de 2018)

#include <stdio.h>
#include <stdlib.h>
#include "image.h" 


/* Rotina para converter imagem colorida em tons de cinza na GPU */ 
__global__ void convert3RGBtoGrayScale(int *input3RGB, int *output1GrayScale, int height, int width) {
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   int j=blockIdx.y*blockDim.y+threadIdx.y;
   if( i<width && j<height ) {   // Importante checar valor do i pois pode acessar fora do tamanho do vetor
      int pos = j*width + i;
      output1GrayScale[ pos ] = 0.299*input3RGB[pos*3] + 0.587*input3RGB[pos*3+1] + 0.114*input3RGB[pos*3+2];
   }
}

/* Programa cria converter imagem colorida em tons de cinza em GPU */
int main(int argc, char** argv) {

   int *d_imageInput, *d_imageOutput;

   cudaError_t error;

   // Estruturas que organizam as imagens PGM
   PGMData *imagemIn  = (PGMData *)malloc(sizeof(PGMData));
   PGMData *imagemOut = (PGMData *)malloc(sizeof(PGMData));

   // Carrega Imagem RGB
   readImage(argv[1],imagemIn);

   // Cria imagem vazia
   createImage(imagemOut, imagemIn->row, imagemIn->col, imagemIn->max_intensity, 1);

   // Aloca vetores na memoria da GPU
   error = cudaMalloc((void **)&d_imageInput,imagemIn->row*imagemIn->col*imagemIn->channels*sizeof(int));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMalloc((void **)&d_imageOutput,imagemIn->row*imagemIn->col*sizeof(int));
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
   convert3RGBtoGrayScale<<<dimGrid,dimBlock>>>(d_imageInput,d_imageOutput,imagemIn->row,imagemIn->col);

   // Retorna valores da memoria da GPU para a CPU
   error = cudaMemcpy(imagemOut->matrix, d_imageOutput, imagemIn->row*imagemIn->col*sizeof(int), cudaMemcpyDeviceToHost);
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
