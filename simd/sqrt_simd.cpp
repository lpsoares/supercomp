// Baseado em: tech.io/playgrounds/283/sse-avx-vectorization
// Atualizado por: Luciano Soares

#include <x86intrin.h> //Extensoes SSE
#include <bits/stdc++.h> //Bibliotecas STD

const int N = 64000000; //Numero de testes
const int V = N/8;      //tamanho vectorizado

//funcao linear
float linear[N];
__attribute__((optimize("no-tree-vectorize"))) //Desliga auto-vectorizacao
inline void normal_sqrt()
{
    for (int i = 0; i < N; ++i)
        linear[i] = sqrtf(linear[i]);
}

__m256 __attribute__((aligned(32))) vectorized[V]; //Vectorized array
inline void avx_sqrt()
{
    for (int i = 0; i < V; ++i)
        vectorized[i] = _mm256_sqrt_ps(vectorized[i]);
  //****** Insira o codigo AVX aqui *******
}

using namespace std;
using namespace std::chrono;

high_resolution_clock::time_point now = high_resolution_clock::now();
#define TIME duration_cast<duration<double>>(high_resolution_clock::now() - now).count()

int main() {
    
    //Inicializacao dos Dados
    for (int i = 0; i < N; ++i) { linear[i] = ((float)i)+ 0.1335f; }
    for (int i = 0; i < V; ++i) {
        for (int v=0;v<8;++v)
         {  vectorized[i][v] = ((float)(i*8+v))+ 0.1335f; }
    }
 
    //Benchmarking de sqrt convencional. 20*64 Milhoes Sqrts
    now = high_resolution_clock::now();
    for (int i = 0; i < 20; ++i)
    normal_sqrt();
    double linear_time = TIME;
    cerr << "Normal sqrtf: "<< linear_time << endl;
 
    //Benchmarking vetorizado por AVX. 20*8*8 Milhoes Sqrts
    now = high_resolution_clock::now();
    for (int i = 0; i < 20; ++i)
    avx_sqrt();
    double avx_time = TIME;
    cerr << "AVX sqrtf : "<<avx_time << endl;
    
    //Converindo valores
    for (int i = 0; i < V; ++i) {
        for (int v=0;v<8;++v)
         { 
           if (abs(linear[i*8+v] - vectorized[i][v]) > 0.00001f)
           {
             cerr << "ERRO: sqrt AVX difere do mesmo calculado de forma linear!";
             cerr << linear[i*8+v]<<" <-> "<<vectorized[i][v]<<endl;
             return -1;
           }
         }
    }
    cout << "Melhora do codigo AVX sobre o Linear: "<< (linear_time/avx_time*100)<<"%" << endl;

    return 0;
}
