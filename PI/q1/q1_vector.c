#include <stdio.h>
#include "random.h"
#include <x86intrin.h>

static long num_trials = 100000000;  // numero de lancamentos aleatórios para o Monte Carlo

int main ()
{
   long i, j;
   long Ncirc;          // armazena a quantidade de lancamentos que cairam dentro os círculo
   double pi;           // armazena o valor final calculado de pi
    
   __m256d VX, VY;      // armazena os valores aleatorios para as coordenadas X e Y
   __m256d X2, Y2, XY;  // armazena os quadrados e soma de quadrados do valores para as coordenadas X e Y

   __m256i TI;          // armazena o total temporario de cada iteracao do loop

   __m256d one   = _mm256_set1_pd(1*1);      // vetor com preenchido com raio do circulo ao quadrado
   __m256i total = _mm256_setzero_si256();   // vetor dos totais iniciado com zeros 

   seed(-1.0, 1.0);     // seed da funcao randomica com faixa de valor

   for(i=0;i<num_trials; i+=4)
   {

      for(j=0;j<4;j++) {
        VX[j] = drandom();
        VY[j] = drandom();
      }

      // A rotina abaixo nao esta vetorizada, faca as modificacoes
      // no codigo para que ela tire proveito da arquitetura vetorial
      // que voce possui no seu computador.
      for(j=0;j<4;j++) {
        X2[j] = VX[j]*VX[j];
        Y2[j] = VY[j]*VY[j];
        XY[j] = X2[j]+Y2[j];
      }

      // Os valores do vetor sao comparados com o valor do raio e 
      // se menores que o raio o resultado da comparacao eh -1
      // caso contrario resulta em zero, um cast para inteiros eh
      // realizado pois os valores em double sao invalidos.
      TI    =  _mm256_castpd_si256(_mm256_cmp_pd(XY, one, _CMP_LE_OS));

      total =  _mm256_add_epi64(total,TI);  // a cada iteracao os dados sao acumulados

    }
    
    Ncirc = -(total[0]+total[1]+total[2]+total[3]); // soma as quatro posicoes do vetor e ajusta sinal

    pi = 4.0 * ((double)Ncirc/(double)num_trials);  // calcula o pi com os dados coletador

    printf("\n %ld trials, pi is %lf ",num_trials, pi);

    return 0;
}