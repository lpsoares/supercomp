// Exemplo para o curso de Super Computacao
// Criado por: Luciano P. Soares

#include <stdio.h>
#include <stdlib.h>

/* Rotina para copiar dois vetores na GPU */ 
__global__ void add(double *a, double *b, double *c, int N) {
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   if(i<N) {   // Importante checar valor do i pois pode acessar fora do tamanho do vetor
      c[i] = a[i] + b[i];
   }
}

/* Programa cria dois vetores e soma eles em GPU */
int main() {

   double *h_a, *h_b, *h_c;
   double *d_a, *d_b, *d_c;
   int    blocksize, i, n;

   cudaError_t error;

   n=1<<28;

   // Aloca vetores na memoria da CPU
   h_a = (double *)malloc(n*sizeof(double));
   h_b = (double *)malloc(n*sizeof(double));
   h_c = (double *)malloc(n*sizeof(double));

   // Preenche os vetores
   for (i = 0; i < n; i++) {
    h_a[i] = (double)i;
    h_b[i] = (double)n-i;
   }

   // Aloca vetores na memoria da GPU
   error = cudaMalloc((void **)&d_a,n*sizeof(double));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMalloc((void **)&d_b,n*sizeof(double));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMalloc((void **)&d_c,n*sizeof(double));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }


   // Copia valores da CPU para a GPU
   error = cudaMemcpy(d_a, h_a, n*sizeof(double), cudaMemcpyHostToDevice);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMemcpy(d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   // Realiza calculo na GPU
   blocksize = 256;
   add<<<((n-1)/256 + 1),blocksize>>>(d_a,d_b,d_c,n);

   // Retorna valores da memoria da GPU para a CPU
   error = cudaMemcpy(h_c, d_c, n*sizeof(double), cudaMemcpyDeviceToHost);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   // Libera memoria da GPU
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   // Exibe um resultado para checar se valores conferem
   for(i=0;i<n;i++) {
      if(!(i%(n/8))) {
         printf("a[%d] + b[%d] = c[%d] => ",i,i,i);
         printf("%6.1f + %6.1f = %6.1f\n",h_a[i],h_b[i],h_c[i]);
      }
   }
   
   // Libera memoria da CPU
   free(h_a);
   free(h_b);
   free(h_c);

   return 0;
}
