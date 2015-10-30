#include <stdio.h>
#define NT 256

__device__ double devResult[2];

__shared__ double shrResult [NT];

extern "C" __global__ void metricCenter(double *xPts, double *yPts, int n)
{
	int thr,size;
	double thrResult, tempResult;
	
	thr = threadIdx.x;
	//rank = blockIdx.x*NT + thr;
	size = gridDim.x*NT;
	thrResult = 0;

	// Calculate the distance from this block's points to one of the other points.
	for(unsigned long long int i = thr; i < n; i += size)
	{
		tempResult = sqrt(((xPts[blockIdx.x] - xPts[i]) * (xPts[blockIdx.x] - xPts[i])) + ((yPts[blockIdx.x] - yPts[i]) * (yPts[blockIdx.x] - yPts[i])));
		// Keep only the point whose distance is maximum from this block's point
		if(tempResult > thrResult)
		{
			thrResult = tempResult;
		}
	}

	shrResult[thr] = thrResult;

	// Reduce the results of all threads in this block
	__syncthreads();
	for(int i = NT/2; i > 0 ; i >>= 1)
	{
		if(thr < i)
		{
			if(shrResult[thr] < shrResult[thr+i])
				shrResult[thr] = shrResult[thr+i];
		}
		__syncthreads();
	}

	// If this is the 1st thread of the block, it will now have the reduced result of this block.
	// Store this block's result in the devResult array at this block's index.
	if (thr == 0)
	{
		devResult[blockIdx.x] = shrResult[0];
	}

}