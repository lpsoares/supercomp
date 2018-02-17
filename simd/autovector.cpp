// Original: https://tech.io/playgrounds/283/sse-avx-vectorization/autovectorization

#include <bits/stdc++.h>
using namespace std::chrono;

int main() {
	const int N = 200000; //Tamanho do Vetor
	const int noTests = 10000; //Numero de testes
	float a[N],b[N],c[N],result[N];
	auto now = high_resolution_clock::now();
	for (int i = 0; i < N; ++i) {
		a[i] =       ((float)i)+ 0.1335f; 
		b[i] = 1.50f*((float)i)+ 0.9383f; 
		c[i] = 0.33f*((float)i)+ 0.1172f; 
	}
	for (int i = 0; i < noTests; ++i) 
		for (int j = 0; j < N; ++j) {
			result[j] = a[j]+b[j]-c[j]+3*(float)i;
		}
	auto end_time = duration_cast<duration<double>>(high_resolution_clock::now() - now).count();
	assert( result[2] == ( 2.0f + 0.1335f)+( 1.50f*2.0f + 0.9383f)-(0.33f*2.0f+0.1172f)+3*(float)(noTests-1));
	std::cout << "Tempo: "<< end_time << "s, N * noTests="<<(N*noTests)<< std::endl;
	return 0;
}


