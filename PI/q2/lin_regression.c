#include <stdlib.h>

double *transpose(double *__X, int rows, int cols) {
    double (*T)[rows] = malloc(sizeof(double) * rows * cols);
    double (*X)[cols] = (double(*)[cols]) __X;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T[j][i] = X[i][j];
        }
    }
    return (double *) T;
}

void compute_XTX(double *__XT, int rows, int cols, double *__XTX) {
    double (*XT)[rows] = (double(*)[rows]) __XT;
    double (*XTX)[cols] = (double(*)[cols]) __XTX;

    
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            XTX[i][j] = 0.0;
            for (int k = 0; k < rows; k++) {
                XTX[i][j] += XT[i][k] * XT[j][k];
            }
        }
    }
}

void matriz_vector_multiply(double *__m1, double *v1, int rows, int cols, double *res) {
    double (*m1)[cols] = (double(*)[cols]) __m1;

    for (int i = 0; i < rows; i++) {
        res[i] = 0;
        for (int j = 0; j < cols; j++) {
            res[i] += m1[i][j] * v1[j];
        }
    }
}

/* v2 -= v1 */
void vector_inplace_sub(double *v1, double *v2, int n) {
    for (int i = 0; i < n; i++) {
        v2[i] -= v1[i];
    }
}

void vector_inplace_square(double *v1, int n) {
    for (int i = 0; i < n; i++) {
        v1[i] *= v1[i];
    }
}

double vector_sum(double *v, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += v[i];
    }
    return sum;
}

void vector_inplace_multiply_scalar(double *v, int n, double s) {
    for (int i = 0; i < n; i++) {
        v[i] *= s;
    }
}


double cost_function(double *X, double *y, double *w, int rows, int cols) {
    static double *temp1 = NULL;
    if (temp1 == NULL) {
        temp1 = malloc(sizeof(double) * rows);
    }
    matriz_vector_multiply(X, w, rows, cols, temp1);
    vector_inplace_sub(y, temp1, rows);
    vector_inplace_square(temp1, rows);
    return vector_sum(temp1, rows);
}

void update_weights(double *X, double *XT, double *y, double *w, double learning_rate, int rows, int cols) {       
    static double *temp1 = NULL, *temp2 = NULL, *temp3 = NULL;
    if (temp1 == NULL) {
        temp1 = malloc(sizeof(double) * cols * cols); /* X^T X */
        temp2 = malloc(sizeof(double) * cols); /* X^T X w */
        temp3 = malloc(sizeof(double) * cols); /* X^T y */
    }
    compute_XTX(XT, rows, cols, temp1);
    matriz_vector_multiply(temp1, w, cols, cols, temp2);
    matriz_vector_multiply(XT, y, cols, rows, temp3);
    vector_inplace_sub(temp3, temp2 ,cols);
    vector_inplace_multiply_scalar(temp2, cols, 2.0*learning_rate/rows);
    vector_inplace_sub(temp2, w, cols);
}