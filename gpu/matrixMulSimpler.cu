// Código Matrix multiplication: C = A * B. Super simplificado.

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void multMat(float *A, float *B, float *C, int wA, int wB)
{

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}



int main(int argc, char **argv)
{

    const int size = 32;
    const int nIter = 100000;

    float *h_A = (float *)malloc(sizeof(float) * size * size);
    float *h_B = (float *)malloc(sizeof(float) * size * size);
    float *h_C = (float *)malloc(sizeof(float) * size * size);
    for (int i = 0; i < size * size; ++i) { h_A[i] =  1.0f; h_B[i] =  1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(float) * size * size);
    cudaMalloc((void **) &d_B, sizeof(float) * size * size);
    cudaMalloc((void **) &d_C, sizeof(float) * size * size);

    cudaMemcpy(d_A, h_A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    dim3 threads(size, size);
    dim3 grid(size / threads.x, size / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    for (int j = 0; j < nIter; j++)
        multMat<<<grid,threads>>>(d_A, d_B, d_C, size, size);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;    
    printf("Time= %2.5f\n",msecPerMatrixMul);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(size * size); i++)
    {
        double abs_err = fabs(h_C[i] - (size * 1.0f));
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/size ;
        if (rel_err > eps)
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], size*1.0f, eps);
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return(0);
}
