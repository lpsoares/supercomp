
// Exemplo de produto escalar

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int main (void){
    cublasHandle_t handle;

    float *devPtrA;
    float *devPtrB;

    float a[3] = {2,0,0};
    float b[3] = {3,0,0};        
    float c;

    cudaMalloc((void**)&devPtrA, 3*sizeof(float));
    cudaMalloc((void**)&devPtrB, 3*sizeof(float));
    
    cublasCreate(&handle);
    
    cublasSetVector(3, sizeof(float), &a, 1, devPtrA, 1);
    cublasSetVector(3, sizeof(float), &b, 1, devPtrB, 1);
    
    cublasSdot(handle, 3, devPtrA, 1, devPtrB, 1, &c);
        
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cublasDestroy(handle);
    
    printf ("%3.0f", c);

    return EXIT_SUCCESS;
}