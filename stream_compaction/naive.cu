#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int pow2dminusone, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= pow2dminusone) {
                int temp = idata[index];
                int temp1 = idata[index - pow2dminusone];
                odata[index] = temp + temp1;
            }
		}

        __global__ void kernToExclusive(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            odata[index] -= idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			const int blockSize = BLOCK_SIZE;
            int padSize = (int)pow(2.f, ilog2ceil(n));
			dim3 gridDim((padSize + blockSize - 1) / blockSize);
			dim3 blockDim(blockSize);
			//std::cout << "padsize " << padSize << std::endl;
			int* device_idata = nullptr;
			int* device_odata = nullptr;
			cudaMalloc((void**)&device_idata, padSize * sizeof(int));
			cudaMalloc((void**)&device_odata, padSize * sizeof(int));
			cudaMemset(device_odata, 0, padSize * sizeof(int));
            cudaMemset(device_idata, 0, padSize * sizeof(int));
            cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("naive init failed!");
			{
				SCOPED_GPU_TIMER
                for (int pow2dminusone = 1; pow2dminusone < padSize; pow2dminusone <<= 1)
                {
                    kernNaiveScan << <gridDim, blockDim >> > (n, pow2dminusone, device_odata, device_idata);
                    cudaMemcpy(device_idata, device_odata, padSize * sizeof(int), cudaMemcpyHostToDevice);
                    checkCUDAError("kernNaiveScan init failed!");
                }
                cudaMemcpy(device_odata+1, device_idata, (n-1) * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemset(device_odata, 0, sizeof(int));
			}
            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            checkCUDAError("cudaMemcpy init failed!");
        }
    }
}
