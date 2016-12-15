#include <iostream>
#include <vector>
#include <random>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>

using std::vector;

#define SIZE 10

int main()
{
  thrust::device_vector<float> d_V1(SIZE);
  thrust::device_vector<float> d_V2(SIZE);
  thrust::device_vector<float> d_V3(SIZE);
  thrust::host_vector<float> h_V1(SIZE);
  thrust::host_vector<float> h_V2(SIZE);
  thrust::host_vector<float> h_V3(SIZE);

  thrust::sequence(h_V1.begin(), h_V1.end(), 1);
  thrust::fill(h_V2.begin(), h_V2.end(), 75);

  std::cout << "-----   V1   -----" << std::endl;
  for(int i = 0; i < SIZE; ++i) std::cout << h_V1[i] << " ";
  std::cout << std::endl;

  std::cout << "-----   V2   -----" << std::endl;
  for(int i = 0; i < SIZE; ++i) std::cout << h_V2[i] << " ";
  std::cout << std::endl;

  d_V1 = h_V1;
  d_V2 = h_V2;

  thrust::transform(d_V1.begin(), d_V1.end(), d_V2.begin(), d_V3.begin(), thrust::minus<float>());

  thrust::copy(d_V3.begin(), d_V3.end(), h_V3.begin());

  std::cout << "-----   V3   -----" << std::endl;
  for(int i = 0; i < SIZE; ++i) std::cout << h_V3[i] << " ";
  std::cout << std::endl;

  return 0;
}
