// Programa exemplo para mostrar funcionamento de sistema multithread.
#include <stdio.h>
#include <omp.h>
int main() {
	#pragma omp parallel
	{
		int n = omp_get_num_threads(); // armazenará o numero de thread
		int t = omp_get_thread_num(); // armazenará o identificador da thread
		printf("Processando na thread %d num total de %d threads\n", t, n);
	}
	return 0;
}

