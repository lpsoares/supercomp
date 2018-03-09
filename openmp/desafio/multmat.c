// Desafio: otimize esse código para calcular o determinante no menor tempo possível 
#include <stdio.h>
#define N 12

// Calculo do cofator 
void getCofactor(double mat[N][N], double temp[N][N], int p, int q, int n) {
    int row,col,i=0,j=0;
    for ( row = 0; row < n; row++) {
        for ( col = 0; col < n; col++) {
            if (row != p && col != q) {
                temp[i][j++] = mat[row][col];
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
 
// Funcao que encontra o determinante de uma matrix
double determinant(double mat[N][N], int n) {
    double det = 0; // inicia com zero

    //  Caso base: se a matriz contem um elemento
    if (n == 1)
        return mat[0][0];
 
    double temp[N][N]; // Armazena os cofatores
 
    double sign = 1;  // Armazena o sinal
    int f;

    // itera na linha
    for (f = 0; f < n; f++) {
        getCofactor(mat, temp, 0, f, n);  // monta cofatores
        det += sign * mat[0][f] * determinant(temp, n - 1); 
        sign = -sign;
    }

    return det;
}
 
// Exibe a matriz
void display(double mat[N][N], int row, int col) {
    int i,j;
    for ( i = 0; i < row; i++) {
        for (j = 0; j < col; j++)
            printf("%2.2f ", mat[i][j]);
        printf("\n");
    }
}

int main (int argc, char *argv[]) {

  int i, j, k=1;
  double a[N][N];   /* matriz A */

  /*** Criando matrizes ***/
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j]= (i==j?(i+j+k++)/10.0:0);

  display(a,N,N);

  printf("Determinante = %f\n", determinant(a, N));

}
