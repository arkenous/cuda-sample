#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>


struct function_object
{
   __device__ float operator () (const float & x)
   {
       return 2.0 * x + 1.0;
   }
};


int main()
{
   /* 1) allcate */
   thrust::host_vector < float > host_vec(3);
   thrust::device_vector < float > device_input(3);
   thrust::device_vector < float > device_output(3);

   /* 2) initialize */
   host_vec[0] = 1.1;
   host_vec[1] = 3.3;
   host_vec[2] = 2.2;

   /* 3) copy host to device */
   thrust::copy(host_vec.begin(), host_vec.end(), device_input.begin());

   /* 4) transform devide_input to device_output */
   thrust::transform(device_input.begin(), device_input.end(), device_output.begin(), function_object());

   /* 5) copy device to host */
   thrust::copy(device_output.begin(), device_output.end(), host_vec.begin());

   std::cout << host_vec[0] << std::endl;
   std::cout << host_vec[1] << std::endl;
   std::cout << host_vec[2] << std::endl;

   return 0;
}
