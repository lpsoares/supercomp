// Programa exemplo para mostrar funcionamento de sistema multithread.
#include <stdio.h>
#include <omp.h>
int main() {
	int acum=0;
	#pragma omp parallel
	{
		int n = omp_get_num_threads(); // armazenará o numero de thread
		int t = omp_get_thread_num(); // armazenará o identificador da thread
		acum+=1;
		printf("Thread %d/%d, acumulando %d\n", t, n, acum);
	}
	printf("Valor final do acumulador %d\n", acum);
	return 0;
}

