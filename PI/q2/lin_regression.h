

double cost_function(double *X, double *y, double *w, int rows, int cols);
void update_weights(double *X, double *XT, double *y, double *w, int rows, int cols, double learning_rate);

double *transpose(double *X, int rows, int cols);