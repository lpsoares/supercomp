#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <vector>


int main() {
    std::vector<double> data;
    double temp;

    while (scanf("%lf", &temp) > 0) {
        data.push_back(temp);
    }

    thrust::device_vector<double> gpu_data(data);
    thrust::device_vector<double> var(data.size());

    double mean = (double) thrust::reduce(gpu_data.begin(), gpu_data.end(), (double) 0, thrust::plus<double>()) / data.size();

    thrust::transform(gpu_data.begin(), gpu_data.end(), thrust::constant_iterator<double>(mean), var.begin(), thrust::minus<double>());
    thrust::transform(var.begin(), var.end(), var.begin(), var.begin(), thrust::multiplies<double>());
    double variance = (double) thrust::reduce(var.begin(), var.end(), (double) 0, thrust::plus<double>()) / data.size();
    printf("Mean: %lf Variance: %lf Size: %d\n", mean, variance, data.size());



}