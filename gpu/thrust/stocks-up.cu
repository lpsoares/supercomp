#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <vector>
#include <thrust/iterator/zip_iterator.h>


typedef thrust::tuple<double, double> D2;
typedef thrust::device_vector<double>::iterator DIter;
typedef thrust::tuple<DIter, DIter> DIter2Tuple;
typedef thrust::zip_iterator<DIter2Tuple> DIter2Iterator;

struct up {
    __host__ __device__
    int operator() (D2 pair) {
        if (thrust::get<0>(pair) < thrust::get<1>(pair)) {
            return 1;
        }
        return 0;
    }
};

int main() {
    std::vector<double> data;
    double temp;

    while (scanf("%lf", &temp) > 0) {
        data.push_back(temp);
    }

    thrust::device_vector<double> gpu_data(data);
    thrust::device_vector<int> is_up(data.size()-1);
    
    DIter2Iterator start = thrust::make_zip_iterator(
            thrust::make_tuple(gpu_data.begin(), gpu_data.begin()+1)
        );
    DIter2Iterator end = thrust::make_zip_iterator(
        thrust::make_tuple(gpu_data.end()-1, gpu_data.end())
    );
    thrust::transform(start, end, is_up.begin(), up());

    int num_up = thrust::reduce(is_up.begin(), is_up.end(), 0, thrust::plus<int>());
    printf("num_up: %d\n", num_up);
}