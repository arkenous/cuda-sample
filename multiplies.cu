#include <iostream>
#include <vector>
#include <random>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

using std::vector;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;

#define SIZE 100000

int main()
{
  clock_t start = clock();

  random_device rnd;
  mt19937 mt;
  mt.seed(rnd());
  uniform_real_distribution<double> real_rnd(0.0, 1.0);

  vector<double> host_w(SIZE, 10.0);
  vector<double> host_input(SIZE);
  vector<double> host_output(SIZE);

  for(int i = 0; i < SIZE; ++i) host_input[i] = real_rnd(mt);

  // allocate device side vector
  thrust::device_vector<double> device_w(SIZE);
  thrust::device_vector<double> device_input(SIZE);
  thrust::device_vector<double> device_output(SIZE);

  // copy host to device
  thrust::copy(host_w.begin(), host_w.end(), device_w.begin());
  thrust::copy(host_input.begin(), host_input.end(), device_input.begin());

  // device_output = device_w * device_input
  clock_t transform_start = clock();
  thrust::transform(device_w.begin(), device_w.end(), device_input.begin(), device_output.begin(), thrust::multiplies<double>());
  clock_t transform_end = clock();

  // copy device to host
  thrust::copy(device_output.begin(), device_output.end(), host_output.begin());

  clock_t end = clock();

  std::cout << "transform time: " << (double)(transform_end - transform_start) / CLOCKS_PER_SEC << std::endl;
  std::cout << "program time: " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

  return 0;
}
