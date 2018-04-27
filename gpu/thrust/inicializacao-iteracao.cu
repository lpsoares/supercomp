#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

int main() {
    thrust::host_vector<double> host(10, 0);
    host[9] = 35;

    printf("Host vector: ");
    for (thrust::host_vector<double>::iterator i = host.begin(); i != host.end(); i++) {
        std::cout << *i << " ";
    }
    printf("\n");

    thrust::device_vector<double> dev(host);
    printf("Device vector: ");
    for (thrust::device_vector<double>::iterator i = dev.begin(); i != dev.end(); i++) {
        std::cout << *i << " ";
    }
    printf("\n");

}
