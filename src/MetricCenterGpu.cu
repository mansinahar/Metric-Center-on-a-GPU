#include <stdio.h>

// Number of threads
#define NT 1024

// Structure to hold the 2D Points
typedef struct
{
	double x;
	double y;
}
point;

// Structure to store the metric center result
typedef struct
{
	double distance;
	int pointIndex;
}
result;

// Function to calculate distance between two points
__device__ double pointDistance(point *aPoint, point *bPoint)
{
	double distance;
	distance = sqrt(((aPoint->x - bPoint->x) * (aPoint->x - bPoint->x)) + ((aPoint->y - bPoint->y) * (aPoint->y - bPoint->y)));
	return distance;
}

// Compare two distances
__device__ int compareDistance(double a, double b)
{
	if(a < b) return -1;
	if(a > b) return 1;
	return 0;
}

// Assign the values of one result struct to another result struct
__device__ void assignResult(result *a, result *b)
{
	a->pointIndex = b->pointIndex;
	a->distance = b->distance;
}

// Function to reduce the block's result
__device__ void reduceBlockResult(result *blockResult, result *newResult)
{
	// Store this block's result in the devResult array at this block's index only if the new result
	// is better than the old result of this block.
	if((blockResult->distance == -100.00 && blockResult->pointIndex == -1) || (compareDistance(blockResult->distance, newResult->distance) == 1))
			{
				assignResult(blockResult, newResult);	
			}
}

// Array holding the result of each thread in a block
__shared__ result shrResult [NT];

// Kernel function to calculate the metric center
extern "C" __global__ void metricCenter(point *pts, result *devResult, int n)
{
	int thr, size, block, noOfBlocks;
	result thrResult, tempResult;
	
	block = blockIdx.x;
	noOfBlocks = gridDim.x;
	thr = threadIdx.x;
	size = NT;

	// Calculate the distance from this block's points to one of the other points.
	for(int i = block; i < n; i += noOfBlocks)
	{
		thrResult.distance = -1.0;
		for(int j = thr; j < n; j += size)
		{
			tempResult.distance = pointDistance(&pts[i], &pts[j]);
			
			// Keep only the point whose distance is maximum from this block's point
			if(compareDistance(tempResult.distance, thrResult.distance) == 1)
			{
				tempResult.pointIndex = i;
				assignResult(&thrResult, &tempResult);
			}
		}
		
		assignResult(&shrResult[thr], &thrResult);

		// Reduce the results of all threads in this block
		__syncthreads();
		for(int m = NT/2; m > 0 ; m >>= 1)
		{
			if(thr < m)
			{
				if(compareDistance(shrResult[thr].distance, shrResult[thr+m].distance) == -1)
				{
					assignResult(&shrResult[thr], &shrResult[thr+m]);
				}	
			}
			__syncthreads();
		}

		// If this is the 1st thread of the block, it will now have the reduced result of this block.
		if (thr == 0)
		{
			reduceBlockResult(&devResult[blockIdx.x], &shrResult[0]);
		}	
	}
}