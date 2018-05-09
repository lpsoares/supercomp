
// Exemplo de operações em vetores

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 5

int main (void){
    cublasHandle_t handle;
    float* devPtrA;
    float* a = 0;
    int i;

    a = (float *)malloc (M * sizeof (float));
   
    for (i = 0; i < M; i++)
            a[i] = (float)rand();

    cudaMalloc((void**)&devPtrA, M*sizeof(float));
    
    cublasCreate(&handle);
    
    cublasSetVector(M, sizeof(float), a, 1, devPtrA, 1);

    int maximo;
    cublasIsamax(handle, M, devPtrA, 1, &maximo);
    printf("max %d\n",maximo);

    int minimo;
    cublasIsamin(handle, M, devPtrA, 1, &minimo);
    printf("min %d\n",minimo);
    
    float soma;
    cublasSasum(handle, M, devPtrA, 1, &soma);
    printf("soma %5.0f\n",soma);

    cudaFree (devPtrA);
    cublasDestroy(handle);

    for (i = 0; i < M; i++)
        printf ("%5.0f  ", a[i]);

    free(a);
    return EXIT_SUCCESS;
}