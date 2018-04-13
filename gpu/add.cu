// Exemplo para o curso de Super Computacao
// Criado por: Luciano P. Soares

#include <stdio.h>
#include <stdlib.h>

/* Rotina para somar dois vetores na GPU */ 
__global__ void add(float *a, float *b, float *c, int N) {
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   if(i<N) {   // Importante checar valor do i pois pode acessar fora do tamanho do vetor
      c[i] = a[i] + b[i];
   }
}

/* Programa cria dois vetores e soma eles em GPU */
int main() {

   float *h_a, *h_b, *h_c;
   float *d_a, *d_b, *d_c;
   int i, n;

   cudaError_t error;

   n=1<<29;

   // Aloca vetores na memoria da CPU
   h_a = (float *)malloc(n*sizeof(float));
   h_b = (float *)malloc(n*sizeof(float));
   h_c = (float *)malloc(n*sizeof(float));

   // Preenche os vetores
   for (i = 0; i < n; i++) {
    h_a[i] = (float)n;
    h_b[i] = (float)n;
   }

   // Aloca vetores na memoria da GPU
   error = cudaMalloc((void **)&d_a,n*sizeof(float));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMalloc((void **)&d_b,n*sizeof(float));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMalloc((void **)&d_c,n*sizeof(float));
   if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }


   // Copia valores da CPU para a GPU
   error = cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   error = cudaMemcpy(d_b, h_b, n*sizeof(float), cudaMemcpyHostToDevice);
   if(error!=cudaSuccess) {
      printf("Memory Copy CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   // Realiza calculo na GPU
   add<<<ceil(n/(float)256),256>>>(d_a,d_b,d_c,n);

   // Retorna valores da memoria da GPU para a CPU
   error = cudaMemcpy(h_c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);
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
