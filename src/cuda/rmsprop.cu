#include <momentum.cuh>

#include <cuda_runtime.h>

__global__ void grad_descent(float *odata, const float *idata, int size)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if(t < size)
	{
		float tmp = odata[t];
		tmp = tmp - (float)LEARNIG_RATE*idata[t];
		odata[t] = tmp;
	}
}