#include <cstdio>
#include <omp.h>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using namespace StreamCompaction::Common;
        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            {
				SCOPED_CPU_TIMER
                if (n == 0)
                {
                    return;
                }
                odata[0] = 0;

                for (int i = 1; i < n; i++)
                {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
        }

        void scanOMP(int n, int* odata, const int* idata)
        {
            int padSize = (int)pow(2.f, ilog2ceil(n));
			const int blockSize = 1024;
            int* temp_odata = new int[padSize];
            int* temp_odata1 = new int[padSize];
            bool bSwap = false;
            if (n == 0)
            {
                return;
            }
            memset(temp_odata, 0, padSize * sizeof(int));
            memset(temp_odata1, 0, padSize * sizeof(int));
            memcpy_s(temp_odata, n * sizeof(int), idata, n * sizeof(int));
            memcpy_s(temp_odata1, n * sizeof(int), idata, n * sizeof(int));
            omp_set_num_threads(CPU_THREADS);
	        {
				SCOPED_CPU_TIMER
                for (int pow2dminusone = 1; pow2dminusone < padSize; pow2dminusone <<= 1)
                {
#pragma omp parallel for
                    for (int i = 0; i < padSize; i+= blockSize)
                    {
                        for (int j = 0; j < blockSize; j++)
                        {
                            int index = i + j;
                            if (index < padSize)
                            {
                                if (index>=pow2dminusone)
                                {
                                    if (bSwap)
                                    {
                                        int temp = temp_odata1[index];
                                        int temp1 = temp_odata1[index - pow2dminusone];
                                        temp_odata[index] = temp + temp1;
                                    }
                                    else
                                    {
                                        int temp = temp_odata[index];
                                        int temp1 = temp_odata[index - pow2dminusone];
                                        temp_odata1[index] = temp + temp1;
                                    }
                                }
                                else
                                {
	                                if (bSwap)
	                                {
										temp_odata[index] = temp_odata1[index];
	                                }else
	                                {
                                        temp_odata1[index] = temp_odata[index];
	                                }
                                }
                            }
						}
                    }
					bSwap = !bSwap;
                }
	        }
            if (bSwap)
            {
                memcpy_s(temp_odata, padSize * sizeof(int), temp_odata1, padSize * sizeof(int));
			}
            memcpy_s(odata + 1, (n-1) * sizeof(int), temp_odata, (n-1) * sizeof(int));
			odata[0] = 0;
            delete[] temp_odata;
            delete[] temp_odata1;
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            SCOPED_CPU_TIMER
            int numElements = 0;
            for (int i = 0;i < n;i++)
            {
                if (idata[i] != 0)
                {
                    odata[numElements] = idata[i];
                    numElements++;
                }
			}
            return numElements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            SCOPED_CPU_TIMER
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
			scan(n, indices, bools);
            for (int i = 0;i<n;i++)
            {
                if (bools[i]==1)
                {
                    odata[indices[i]] = idata[i];
                }
            }
            return indices[n-1];
        }
    }
}
