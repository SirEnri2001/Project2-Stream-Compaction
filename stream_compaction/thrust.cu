#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* device_idata = nullptr;
            int* device_odata = nullptr;
            cudaMalloc((void**)&device_idata, n * sizeof(int));
            cudaMalloc((void**)&device_odata, n * sizeof(int));
            cudaMemset(device_odata, 0, n * sizeof(int));
            cudaMemset(device_idata, 0, n * sizeof(int));
            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            //cudaMemcpy(device_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            thrust::device_ptr<int> dev_ptr_idata(device_idata);
            thrust::device_ptr<int> dev_ptr_odata(device_odata);
            {
                SCOPED_GPU_TIMER
                thrust::exclusive_scan(dev_ptr_idata, dev_ptr_idata + n, dev_ptr_odata);
                checkCUDAError("Thrust scan failed");
            }
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
