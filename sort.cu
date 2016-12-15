#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16;

int main() {
  // allocate
  thrust::host_vector<float> host_vec(3);
  thrust::device_vector<float> device_vec(3);

  // initialize
  host_vec[0] = 1.1;
  host_vec[1] = 3.3;
  host_vec[2] = 2.2;

  // copy host to device
  thrust::copy(host_vec.begin(), host_vec.end(), device_vec.begin());

  // sort
  thrust::sort(device_vec.begin(), device_vec.end());

  // copy device to host
  thrust::copy(device_vec.begin(), device_vec.end(), host_vec.begin());

  std::cout << host_vec[0] << std::endl;
  std::cout << host_vec[1] << std::endl;
  std::cout << host_vec[2] << std::endl;
  
  return 0;
}
