#include <stdio.h>
#include <stdlib.h>

#include "lin_regression.h"

void read_input_data(double **_mtr, int *rows, int *cols) {
    int r, c;
    double *mtr;
    scanf("%d %d", &r, &c);
    mtr = malloc(sizeof(double) * r * c);
    for (int i = 0, k = 0; i < r; i++) {
        for (int j = 0; j < c; j++, k++) {  
            scanf("%lf", &mtr[k]);
        }
    }
    *rows = r;
    *cols = c;
    *_mtr = mtr;
}

double *read_responses(int cols) {
    double *y;
    y = malloc(sizeof(double) * cols);
    for (int i = 0; i < cols; i++) {
        scanf("%lf", &y[i]);
    }
    return y;
}

int main(int argc, char **argv) {
    int rows, cols, epochs;
    double *X, *XT, *y, learning_rate;
    double *w, last_cost;
    read_input_data(&X, &rows, &cols);
    XT = transpose(X, rows, cols);
    y = read_responses(rows);

    w = malloc(sizeof(double) * cols);

    learning_rate = atof(argv[2]);
    epochs = atoi(argv[1]);

    printf("Learning rate: %lf\n", learning_rate);

    printf("Initial cost %lf\n", cost_function(X, y, w, rows, cols));
    
    for (int k = 0; k < epochs; k++) {
        update_weights(X, XT, y, w, rows, cols, learning_rate);
        if (k % 100 == 0) {
            printf("Cost function: %lf\n", cost_function(X, y, w, rows, cols));
        }
    }

    printf("Solution: ");
    for (int i = 0; i < cols; i++) {
        printf("%lf ", w[i]);
    }
    printf("\nCost: %lf\n", cost_function(X, y, w, rows, cols));

    return 0;
}