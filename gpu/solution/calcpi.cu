// Baseado em: https://docs.nvidia.com/cuda/curand/

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>
#include "cublas_v2.h"

#include <iostream>
#include <iomanip>

struct estimate_pi : 
    public thrust::unary_function<unsigned int, float>
{
  __device__
  float operator()(unsigned int thread_id) {
    curandState s;
    float sum = 0;
    unsigned int N = 1<<16; // numero de amostras por thread

    //cublasHandle_t handle;
    //cublasCreate(&handle);

    curand_init(thread_id, 0, 0, &s);  // use como seed o identificador da chamada da funcao

    for(unsigned int i = 0; i < N; ++i) { // calcule para um quarto de circulo N vezes
      float x = curand_uniform(&s);
      float y = curand_uniform(&s);

      float dist = sqrtf(x*x + y*y); // calcule a distancia ate a origem

      if(dist <= 1.0f) // se cair no circulo adicione 1
        sum += 1.0f;
    }

    sum *= 4.0f; // multiplique por 4 para ter a área toda do círculo
    //cublasDestroy(handle);

    return sum / N;  // divida pelo numero de amostras
  }
};

int main(void) {

  int M = 1<<20;

  float estimate = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(M),
        estimate_pi(),
        0.0f,
        thrust::plus<float>());

  estimate /= M;

  std::cout << std::setprecision(7);
  std::cout << "pi e' aproximadamente ";
  std::cout << estimate << std::endl;
  return 0;
}