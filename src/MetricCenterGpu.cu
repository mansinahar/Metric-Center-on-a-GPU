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

__device__ result devResult[3];

__shared__ result shrResult [NT];

extern "C" __global__ void metricCenter(point *pts, int n)
{
	int thr,size;
	result thrResult, tempResult;
	
	thr = threadIdx.x;
	//rank = blockIdx.x*NT + thr;
	size = NT;
	thrResult.distance = 0.0;

	// Calculate the distance from this block's points to one of the other points.
	for(unsigned long long int i = thr; i < n; i += size)
	{
		printf("In for loop of block %d\n", blockIdx.x);
		tempResult.distance = sqrt(((pts[blockIdx.x].x - pts[i].x) * (pts[blockIdx.x].x - pts[i].x)) + ((pts[blockIdx.x].y - pts[i].y) * (pts[blockIdx.x].y - pts[i].y)));
		// Keep only the point whose distance is maximum from this block's point
		if(tempResult.distance > thrResult.distance)
		{
			thrResult.distance = tempResult.distance;
			thrResult.pointIndex = blockIdx.x;
			printf("Block %d's new distance is now %f\n", blockIdx.x, thrResult.distance);
		}
	}

	shrResult[thr].distance = thrResult.distance;
	shrResult[thr].pointIndex = thrResult.pointIndex;

	// Reduce the results of all threads in this block
	__syncthreads();
	for(int i = NT/2; i > 0 ; i >>= 1)
	{
		if(thr < i)
		{
			if(shrResult[thr].distance < shrResult[thr+i].distance)
			{
				shrResult[thr].distance = shrResult[thr+i].distance;
				shrResult[thr].pointIndex = shrResult[thr+i].pointIndex;
			}	
		}
		__syncthreads();
	}

	// If this is the 1st thread of the block, it will now have the reduced result of this block.
	// Store this block's result in the devResult array at this block's index.
	if (thr == 0)
	{
		devResult[blockIdx.x].distance = shrResult[0].distance;
		devResult[blockIdx.x].pointIndex = shrResult[0].pointIndex;
	}

}