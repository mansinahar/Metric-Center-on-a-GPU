#include <stdio.h>
#define NT 256

typedef struct
{
	double x;
	double y;
}
point;

typedef struct
{
	double distance;
	int pointIndex;
}
result;

__device__ result devResult[20];

__shared__ result shrResult [NT];

extern "C" __global__ void metricCenter(point *pts, int n)
{
	int thr, size, block, noOfBlocks;
	result thrResult, tempResult;
	
	block = blockIdx.x;
	noOfBlocks = gridDim.x;
	thr = threadIdx.x;
	size = NT;

	// Calculate the distance from this block's points to one of the other points.
	for(unsigned long long int i = block; i < n; i += noOfBlocks)
	{
		thrResult.distance = -1.0;
		for(unsigned long long int j = thr; j < n; j += size)
		{
			tempResult.distance = sqrt(((pts[i].x - pts[j].x) * (pts[i].x - pts[j].x)) + ((pts[i].y - pts[j].y) * (pts[i].y - pts[j].y)));
			// Keep only the point whose distance is maximum from this block's point
			if(tempResult.distance > thrResult.distance)
			{
				thrResult.distance = tempResult.distance;
				thrResult.pointIndex = i;
			}
		}
		
		shrResult[thr].distance = thrResult.distance;
		shrResult[thr].pointIndex = thrResult.pointIndex;

		// Reduce the results of all threads in this block
		__syncthreads();
		for(int m = NT/2; m > 0 ; m >>= 1)
		{
			if(thr < m)
			{
				if(shrResult[thr].distance < shrResult[thr+m].distance)
				{
					shrResult[thr].distance = shrResult[thr+m].distance;
					shrResult[thr].pointIndex = shrResult[thr+m].pointIndex;
				}	
			}
			__syncthreads();
		}

		// If this is the 1st thread of the block, it will now have the reduced result of this block.
		// Store this block's result in the devResult array at this block's index.
		if (thr == 0)
		{
			if((devResult[blockIdx.x].distance == -100.00 && devResult[blockIdx.x].pointIndex == -1) || (devResult[blockIdx.x].distance > shrResult[0].distance))
			{
				devResult[blockIdx.x].distance = shrResult[0].distance;
				devResult[blockIdx.x].pointIndex = shrResult[0].pointIndex;	
			}
		}	
	}
}