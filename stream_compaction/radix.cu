#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

namespace Radix
{
	__global__ void calculateBAndEArray(int n, int bit, int* bArray, int* eArray, const int* idata)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) return;
		bArray[index] = (idata[index] >> bit) & 1;
		eArray[index] = 1 - bArray[index];
	}

	__global__ void calculateTArray(int n, int* tArray, const int* eScanArray, int totalE)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) return;
		tArray[index] = index - eScanArray[index] + totalE;
	}

	__global__ void calculateNewIndex(int n, int* b, int* e, int* f, int* t, int* out_d)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) return;
		int b_i = b[index];
		out_d[index] = b_i * t[index] + (1 - b_i) * f[index];
	}

	__global__ void scatter(int n, int* odata, const int* idata, const int* d)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) return;
		odata[d[index]] = idata[index];
	}

	void radixDevice(int n, int* device_odata, int* device_idata, int totalBits)
	{
		int blockSize = 512;
		int numBlocks = (n + blockSize - 1) / blockSize;
		// create bArray and eArray for device
		int* device_bArray = nullptr;
		int* device_eArray = nullptr;
		int* device_eScanArray = nullptr;
		int* device_dArray = nullptr;
		int* device_tArray = nullptr;
		cudaMalloc((void**)&device_bArray, n * sizeof(int));
		cudaMalloc((void**)&device_eArray, n * sizeof(int));
		cudaMalloc((void**)&device_tArray, n * sizeof(int));
		cudaMalloc((void**)&device_dArray, n * sizeof(int));
		cudaMalloc((void**)&device_eScanArray, n * sizeof(int));
		checkCUDAError("cudaMalloc for radix failed!");
		{
			SCOPED_GPU_TIMER
			for (int bit = 0; bit < totalBits; bit++)
			{
				calculateBAndEArray << <numBlocks, blockSize >> > (n, bit, device_bArray, device_eArray, device_idata);
				checkCUDAError("calculate bArray and eArray failed!");
				StreamCompaction::MoreEfficient::scanDevice(n, device_eScanArray, device_eArray);
				// get total number of 0s
				int totalE = 0;
				cudaMemcpy(&totalE, &device_eScanArray[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
				int lastE = 0;
				cudaMemcpy(&lastE, &device_eArray[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
				totalE += lastE;
				//std::cout << "Total number of 0s: " << totalE << std::endl;
				// calculate tArray
				calculateTArray << <numBlocks, blockSize >> > (n, device_tArray, device_eScanArray, totalE);
				checkCUDAError("calculate tArray failed!");
				// calculate new index
				calculateNewIndex << <numBlocks, blockSize >> > (n, device_bArray, device_eArray, device_eScanArray, device_tArray, device_dArray);
				checkCUDAError("calculate new index failed!");
				// scatter
				scatter << <numBlocks, blockSize >> > (n, device_odata, device_idata, device_dArray);
				checkCUDAError("scatter failed!");
				cudaMemcpy(device_idata, device_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
		}		
		// free memory
		cudaFree(device_bArray);
		cudaFree(device_eArray);
		cudaFree(device_eScanArray);
		cudaFree(device_tArray);
		cudaFree(device_dArray);
	}

	void radix(int n, int* odata, const int* idata)
	{
		// create device_odata and device_idata
		int* device_odata = nullptr;
		int* device_idata = nullptr;
		cudaMalloc((void**)&device_odata, n * sizeof(int));
		cudaMalloc((void**)&device_idata, n * sizeof(int));
		checkCUDAError("cudaMalloc device_odata and device_idata failed!");
		cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy to device_idata failed!");
		// for each bit
		radixDevice(n, device_odata, device_idata, 16);
		// copy device_odata to odata
		cudaMemcpy(odata, device_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy to odata failed!");
		// free memory
		cudaFree(device_odata);
		cudaFree(device_idata);
	}
}