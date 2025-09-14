#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int pow2dminusone, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;
            if (index >= pow2dminusone) {
                odata[index] = odata[index - pow2dminusone] + odata[index];
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
			const int blockSize = 256;
			dim3 gridDim((n + blockSize - 1) / blockSize);
			dim3 blockDim(blockSize);
            int padSize = (int)pow(2.f, ilog2ceil(n));
			int* device_idata = nullptr;
			int* device_odata = nullptr;
			cudaMalloc((void**)&device_idata, padSize * sizeof(int));
			cudaMalloc((void**)&device_odata, padSize * sizeof(int));
			cudaMemset(device_odata, 0, padSize * sizeof(int));
            cudaMemset(device_idata, 0, padSize * sizeof(int));
			cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int pow2dminusone = 1; pow2dminusone < padSize; pow2dminusone <<= 1)
            {
                kernNaiveScan << <gridDim, blockDim >> > (n, pow2dminusone, device_odata, device_idata);
            }
			kernToExclusive << <gridDim, blockDim >> > (n, device_odata, device_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
