#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>

#include "termcolor.hpp"


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define SCOPED_CPU_TIMER StreamCompaction::Common::ScopedCpuTimer timer##__FUNCTION__(__FUNCTION__, __LINE__);
#define SCOPED_GPU_TIMER StreamCompaction::Common::ScopedGpuTimer timer##__FUNCTION__(__FUNCTION__, __LINE__);

#define BLOCK_SIZE 1024

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

namespace StreamCompaction {
    namespace Common {
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

        class ScopedCpuTimer
        {
            using time_point_t = std::chrono::high_resolution_clock::time_point;
            time_point_t time_start_cpu;
            time_point_t time_end_cpu;
            std::string ScopeName;
            int line;
        public:
			ScopedCpuTimer(std::string InScopeName, int lineNumber) : ScopeName(InScopeName), line(lineNumber)
            {
                time_start_cpu = std::chrono::high_resolution_clock::now();
            }
            ~ScopedCpuTimer()
            {
                time_end_cpu = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
				double milli = duro.count();
                if (milli<1.0)
                {
					std::cout <<termcolor::blue << "[CPU] " << termcolor::reset<< ScopeName << " at line "<<line<< " took " << milli * 1000.0 << " us" << std::endl;
                }else
                {
                    std::cout << termcolor::blue << "[CPU] " << termcolor::reset << ScopeName << " at line " << line << " took " << milli << " ms" << std::endl;
                }
			}
        };

        class ScopedGpuTimer
        {
        public:
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
			std::string ScopeName;
            int line;
			ScopedGpuTimer(std::string InScopeName, int lineNumber) : ScopeName(InScopeName), line(lineNumber)
            {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
            }
            ~ScopedGpuTimer()
            {
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float milli = 0;
                cudaEventElapsedTime(&milli, start, stop);
                if (milli < 1.0)
                {
                    std::cout << termcolor::yellow << "[GPU] " << termcolor::reset << ScopeName << " at line " << line << " took " << milli * 1000.0 << " us" << std::endl;
                }
                else
                {
                    std::cout << termcolor::yellow << "[GPU] " << termcolor::reset << ScopeName << " at line " << line << " took " << milli << " ms" << std::endl;
                }
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
			}
        };


        /**
        * This class is used for timing the performance
        * Uncopyable and unmovable
        *
        * Adapted from WindyDarian(https://github.com/WindyDarian)
        */
        class PerformanceTimer
        {
        public:
            struct TimeElapse
            {
                double time;
				const char* unit;
            };

            PerformanceTimer()
            {
                cudaEventCreate(&event_start);
                cudaEventCreate(&event_end);
            }

            ~PerformanceTimer()
            {
                cudaEventDestroy(event_start);
                cudaEventDestroy(event_end);
            }

            void startCpuTimer(bool useMicrosecs = false)
            {
                if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
                cpu_timer_started = true;
				outputMicrosec = useMicrosecs;

                time_start_cpu = std::chrono::high_resolution_clock::now();
            }

            void endCpuTimer()
            {
                time_end_cpu = std::chrono::high_resolution_clock::now();

                if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

                std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
                prev_elapsed_time_cpu_milliseconds =
                    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());
                if (outputMicrosec)
                {
					prev_elapsed_time_cpu_milliseconds *= 1000.0f;
                }
                cpu_timer_started = false;
            }

            void startGpuTimer(bool useMicrosecs = false)
            {
                if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
                gpu_timer_started = true;
                outputMicrosec = useMicrosecs;

                cudaEventRecord(event_start);
            }

            void endGpuTimer()
            {
                cudaEventRecord(event_end);
                cudaEventSynchronize(event_end);

                if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

                cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
                if (outputMicrosec)
                {
					prev_elapsed_time_gpu_milliseconds *= 1000.0f;
                }
                gpu_timer_started = false;
            }

            TimeElapse getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
            {
                return { prev_elapsed_time_cpu_milliseconds, unit() };
            }

            TimeElapse getGpuElapsedTimeForPreviousOperation() //noexcept
            {
                return { prev_elapsed_time_gpu_milliseconds, unit() };
            }

            const char* unit()
            {
                return outputMicrosec ? "us" : "ms";
			}

            // remove copy and move functions
            PerformanceTimer(const PerformanceTimer&) = delete;
            PerformanceTimer(PerformanceTimer&&) = delete;
            PerformanceTimer& operator=(const PerformanceTimer&) = delete;
            PerformanceTimer& operator=(PerformanceTimer&&) = delete;

        private:
            cudaEvent_t event_start = nullptr;
            cudaEvent_t event_end = nullptr;

            using time_point_t = std::chrono::high_resolution_clock::time_point;
            time_point_t time_start_cpu;
            time_point_t time_end_cpu;

            bool cpu_timer_started = false;
            bool gpu_timer_started = false;
            bool outputMicrosec = false;

            float prev_elapsed_time_cpu_milliseconds = 0.f;
            float prev_elapsed_time_gpu_milliseconds = 0.f;
        };
    }
}
