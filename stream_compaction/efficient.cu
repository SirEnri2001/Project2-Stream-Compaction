#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 32

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scanByBlock(int n, int *odata, const int *idata) {
            extern __shared__ int temp[BLOCK_SIZE]; // allocated on invocation
            int id = threadIdx.x + BLOCK_SIZE * blockIdx.x;
            int tid = threadIdx.x;
            if (id >= n) {
                return;
			}
			int blockSize = BLOCK_SIZE;
            temp[tid] = idata[id]; // load input into shared memory
            for (int d = 2; d<blockSize;d<<=1)
            {
                __syncthreads();
	            if (tid%d==d-1)
	            {
					temp[tid] += temp[tid - (d / 2)];
	            }
            }
            __syncthreads();
			odata[id] = temp[tid];
		}

        __global__ void upsweep(int n, int stride, int *data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index % stride == 0) {
                data[index + stride-1] += data[index + (stride / 2)-1];
            }
		}

        __global__ void downsweep(int n, int stride, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index % stride == 0) {
                int t = data[index + (stride / 2) - 1];
                data[index + (stride / 2) - 1] = data[index + stride - 1];
                data[index + stride - 1] += t;
            }
        }

        __global__ void addArray(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] += idata[index];
		}

        __global__ void getBoolArray(int n, int* bools, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (idata[index] != 0) {
                bools[index] = 1;
            }
            else {
                bools[index] = 0;
            }
		}


        void scan_1(int n, int* odata, const int* idata, bool startTimer) {
            const int blockSize = BLOCK_SIZE;
            dim3 gridDim((n + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
            int padSize = (int)pow(2.f, ilog2ceil(n));
            int* device_idata = nullptr;
            int* device_odata = nullptr;
            int* device_blockSum = nullptr;
            checkCUDAError("before");
            cudaMalloc((void**)&device_idata, n * sizeof(int));
            cudaMalloc((void**)&device_odata, n * sizeof(int));
            cudaMalloc((void**)&device_blockSum, BLOCK_SIZE * sizeof(int));
            cudaMemset(device_odata, 0, padSize * sizeof(int));
            cudaMemset(device_idata, 0, padSize * sizeof(int));
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cuda init scan");
            if (startTimer)
            {
                timer().startGpuTimer();

            }
            scanByBlock << <gridDim, blockDim >> > (n, device_odata, device_idata);
            checkCUDAError("scanByBlock");
            //for (int stride = 2; stride <= padSize; stride <<= 1)
            //{
            //    upsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
            //}

            //cudaMemset(device_odata+padSize-1, 0, sizeof(int));

            //for (int stride = padSize; stride >= 2; stride >>= 1)
            //{
            //    downsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
            //}


            if (startTimer)
            {
                timer().endGpuTimer();
            }

            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            //odata[n - 1] = odata[n - 2] + idata[n - 1];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool startTimer) {
            const int blockSize = BLOCK_SIZE;
            dim3 gridDim((n + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
            int padSize = (int)pow(2.f, ilog2ceil(n));
            int* device_idata = nullptr;
            int* device_odata = nullptr;
            checkCUDAError("before");
            cudaMalloc((void**)&device_idata, padSize * sizeof(int));
            cudaMalloc((void**)&device_odata, padSize * sizeof(int));
            cudaMemset(device_odata, 0, padSize * sizeof(int));
            cudaMemset(device_idata, 0, padSize * sizeof(int));
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cuda init scan");
            if (startTimer)
            {
                timer().startGpuTimer();
	            
            }
            for (int stride = 2; stride <= padSize; stride <<= 1)
            {
                upsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
                checkCUDAError("upsweep");
            }

            cudaMemset(device_odata+padSize-1, 0, sizeof(int));

            for (int stride = padSize; stride >= 2; stride >>= 1)
            {
                downsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
                checkCUDAError("downsweep");
            }


            if (startTimer)
            {
                timer().endGpuTimer();
            }

            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(device_odata);
            cudaFree(device_idata);
            checkCUDAError("cudaFree");
			//odata[n - 1] = odata[n - 2] + idata[n - 1];
        }

        void scanDevice(int n, int* device_odata) {
            const int blockSize = 256;
            dim3 gridDim((n + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
            int padSize = (int)pow(2.f, ilog2ceil(n));
            for (int stride = 2; stride <= padSize; stride <<= 1)
            {
                upsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
            }

            cudaMemset(device_odata + padSize - 1, 0, sizeof(int));

            for (int stride = padSize; stride >= 2; stride >>= 1)
            {
                downsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
            }
        }

        __global__ void compactByIndicies(int n, int* odata, const int* idata, const int* indices) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index==n-1 && idata[index]!=0)
            {
	            odata[indices[index]] = idata[index];
				return;
            }
            if (indices[index]!=indices[index+1])
            {
                odata[indices[index]] = idata[index];
            }
		}

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            const int blockSize = 256;
            dim3 gridDim((n + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
            int padSize = (int)pow(2.f, ilog2ceil(n));
            int* device_idata = nullptr;
            int* device_odata = nullptr;
			int* device_bool = nullptr;
            cudaMalloc((void**)&device_idata, padSize * sizeof(int));
            cudaMalloc((void**)&device_odata, padSize * sizeof(int));
            cudaMalloc((void**)&device_bool, padSize * sizeof(int));
            cudaMemset(device_odata, 0, padSize * sizeof(int));
            cudaMemset(device_idata, 0, padSize * sizeof(int));
            cudaMemset(device_bool, 0, padSize * sizeof(int));
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
			getBoolArray << <gridDim, blockDim >> > (padSize, device_bool, device_idata);
            scanDevice(padSize, device_bool);
			compactByIndicies << <gridDim, blockDim >> > (n, device_odata, device_idata, device_bool);
            timer().endGpuTimer();
			cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			int numCompacted = 0;
			cudaMemcpy(&numCompacted, device_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n-1]!=0)
            {
                numCompacted++;
            }
            cudaFree(device_idata);
			cudaFree(device_odata);
			cudaFree(device_bool);
            checkCUDAError("compact");
            return numCompacted;
        }
    }
}
