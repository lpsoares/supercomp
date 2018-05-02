#include <cstdio>
#include <cstdlib>
#include "image.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// para compilar use:
// gcc -I.. imagem-thrust.cu ../image.cu -o image-thrust


int main(int argc, char *argv[]) {
    PGMData *imagemIn  = (PGMData *)malloc(sizeof(PGMData));
    readImage(argv[1],imagemIn);

    int sz = imagemIn->row*imagemIn->col*imagemIn->channels;
    thrust::device_vector<int> imagemInGPU (sz);
    thrust::copy(imagemIn->matrix, imagemIn->matrix + sz, imagemInGPU.begin());

}