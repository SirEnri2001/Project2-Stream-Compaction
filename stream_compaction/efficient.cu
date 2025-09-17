#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

namespace StreamCompaction {
    namespace MoreEfficient
    {
        __global__ void scanByBlock(int n, int* odata, const int* idata, int* aux) {
            extern __shared__ int temp[BLOCK_SIZE]; // allocated on invocation
            int id = threadIdx.x + BLOCK_SIZE * blockIdx.x;
            int tid = threadIdx.x;
            int blockSize = BLOCK_SIZE;
            temp[tid] = idata[id];

            // up-sweep
            for (int d = 2; d <= blockSize ; d <<= 1)
            {
                __syncthreads();
                if (tid % d == d - 1)
                {
                    temp[tid] += temp[tid - (d / 2)];
                }
            }
            __syncthreads();
            if (tid == blockSize-1)
            {
                aux[blockIdx.x] = temp[tid];
                temp[tid] = 0;
            }

            // down-sweep
            for (int d = blockSize; d >= 2; d >>= 1)
            {
                __syncthreads();
                if (tid % d == d - 1)
                {
                    int t = temp[tid - (d / 2)];
                    temp[tid - (d / 2)] = temp[tid];
                    temp[tid] += t;
                }
            }

            __syncthreads();
            odata[id] = temp[tid];
        }

        __global__ void addAux(int n, int* odata, const int* aux) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (blockIdx.x > 0)
            {
                odata[index] += aux[blockIdx.x];
            }
        }

        void scanDevice(int n, int* device_odata, int* device_idata, int recursive_depth)
        {
            const int blockSize = BLOCK_SIZE;
            std::unique_ptr<int> a = std::make_unique<int>();
            dim3 gridDim((n + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
            int* device_aux = nullptr;
            int auxSize = (int)pow(2.f, ilog2ceil(gridDim.x));
			//std::cout << "auxSize: " << auxSize << std::endl;
            auxSize = (auxSize + blockSize - 1) / blockSize*blockSize;
            //std::cout << "pad auxSize: " << auxSize << std::endl;
            cudaMalloc((void**)&device_aux, auxSize * sizeof(int));
            //std::cout << "n: " << n << std::endl;

            scanByBlock << <gridDim, blockDim >> > (n, device_odata, device_idata, device_aux);
            checkCUDAError("scanByBlock");
			//std::cout << "recursive_depth: " << recursive_depth << std::endl;
            if (gridDim.x>1)
            {
                scanDevice(auxSize, device_aux, device_aux, recursive_depth+1);
            }
            addAux << <gridDim, blockDim >> > (n, device_odata, device_aux);
			checkCUDAError("addAux");
			cudaFree(device_aux);
        }

        void scan(int n, int* odata, const int* idata) {
            int* device_idata = nullptr;
            int* device_odata = nullptr;
            int paddedSize = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            checkCUDAError("before");
            cudaMalloc((void**)&device_idata, paddedSize * sizeof(int));
            cudaMalloc((void**)&device_odata, paddedSize * sizeof(int));
            cudaMemset(device_odata, 0, paddedSize * sizeof(int));
            cudaMemset(device_idata, 0, paddedSize * sizeof(int));

            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cuda init scan");
            {
                SCOPED_GPU_TIMER
                scanDevice(paddedSize, device_odata, device_idata);
            }
            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    namespace Efficient {

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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
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
            {
	            SCOPED_GPU_TIMER
                for (int stride = 2; stride <= padSize; stride <<= 1)
                {
                    upsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
                    checkCUDAError("upsweep");
                }

                cudaMemset(device_odata + padSize - 1, 0, sizeof(int));

                for (int stride = padSize; stride >= 2; stride >>= 1)
                {
                    downsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
                    checkCUDAError("downsweep");
                }
            }
            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(device_odata);
            cudaFree(device_idata);
            checkCUDAError("cudaFree");
			//odata[n - 1] = odata[n - 2] + idata[n - 1];
        }

        void scanDevice(int n, int* device_odata) {
            const int blockSize = BLOCK_SIZE;
            int padSize = (int)pow(2.f, ilog2ceil(n));
            dim3 gridDim((padSize + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
            for (int stride = 2; stride <= padSize; stride <<= 1)
            {
                upsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
                checkCUDAError("upsweep");
            }

            cudaMemset(device_odata + padSize - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset");

            for (int stride = padSize; stride >= 2; stride >>= 1)
            {
                downsweep << <gridDim, blockDim >> > (padSize, stride, device_odata);
                checkCUDAError("downsweep");
            }
        }

        __global__ void compactByIndicies(int n, int* odata, const int* idata, const int* indices) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int in_index = indices[index];
			int in_data = idata[index];

            if (index>=n-1)
            {
                if (in_data==0)
                {
                    odata[in_index] = in_data;
                }
				return;
            }
            int in_index_plus_one = indices[index + 1];
            if (in_index != in_index_plus_one)
            {
                odata[in_index] = in_data;
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
            const int blockSize = BLOCK_SIZE;
            int padSize = (int)pow(2.f, ilog2ceil(n));
            dim3 gridDim((padSize + blockSize - 1) / blockSize);
            dim3 blockDim(blockSize);
			std::cout << "padSize: " << padSize << std::endl;
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
            checkCUDAError("compact init");
            {
	            SCOPED_GPU_TIMER
                getBoolArray << <gridDim, blockDim >> > (padSize, device_bool, device_idata);
                checkCUDAError("getBoolArray");
                scanDevice(padSize, device_bool);
                checkCUDAError("scanDevice");
                compactByIndicies << <gridDim, blockDim >> > (n, device_odata, device_idata, device_bool);
                checkCUDAError("compactByIndicies");
            }
			cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy 1");
			int numCompacted = 0;
			cudaMemcpy(&numCompacted, device_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy 2");
            if (idata[n-1]!=0)
            {
                numCompacted++;
            }
            cudaFree(device_idata);
			cudaFree(device_odata);
			cudaFree(device_bool);
            checkCUDAError("compact finish");
            return numCompacted;
        }
    }
}
