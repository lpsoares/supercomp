
// Exemplo de calculo de normal

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int main (void){
    cublasHandle_t handle;
    float *devPtrA;
    
    float a[3] = {1,1,1};
    float b;
    
    cudaMalloc((void**)&devPtrA, 3*sizeof(float));
    
    cublasCreate(&handle);
    
    cublasSetVector(3, sizeof(float), &a, 1, devPtrA, 1);
    
    cublasSnrm2(handle, 3, devPtrA, 1, &b);
    
    cudaFree(devPtrA);
    cublasDestroy(handle);

    printf ("%f", b);
    
    return EXIT_SUCCESS;
}