#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata, bool setTimer) {
            if (setTimer)
            {
                timer().startCpuTimer(true);
            }
            if (n==0)
            {
                return;
            }
			odata[0] = 0;
            for (int i = 1;i<n;i++)
            {
				odata[i] = odata[i - 1] + idata[i-1];
            }
            if (setTimer)
            {
                timer().endCpuTimer();
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer(true);
            int numElements = 0;
            for (int i = 0;i < n;i++)
            {
                if (idata[i] != 0)
                {
                    odata[numElements] = idata[i];
                    numElements++;
                }
			}
            timer().endCpuTimer();
            return numElements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
			int* bools = new int[n];
			int* indices = new int[n];
            for (int i = 0; i < n; i++)
            {
	            if (idata[i]!=0)
	            {
                    bools[i] = 1;
	            }else
	            {
                    bools[i] = 0;
	            }
            }
			scan(n, indices, bools, false);
            for (int i = 0;i<n;i++)
            {
                if (bools[i]==1)
                {
                    odata[indices[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return indices[n-1];
        }
    }
}
