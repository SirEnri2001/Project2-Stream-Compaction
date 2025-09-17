CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Henry Han
  * https://github.com/sirenri2001
  * https://www.linkedin.com/in/henry-han-a832a6284/
* Tested on: Windows 11 Pro 24H2, i7-9750H @ 2.60GHz 16GB, RTX 2070 Max-Q

# Extra Credit

- Completed Radix Sort & Testing
- Completed Optimizing with Shared Memory & Hardware (see More Effecient Implementation below)

# Test Result

```
****************
** SCAN TESTS **
****************
==== cpu scan, power-of-two ====
[CPU] StreamCompaction::CPU::scan at line 17 took 108.015 ms
==== cpu scan, non-power-of-two ====
[CPU] StreamCompaction::CPU::scan at line 17 took 108.221 ms
    passed
==== gpu radix sort ====
[GPU] Radix::radixDevice at line 56 took 314.12 ms
    passed
==== naive scan, power-of-two ====
[GPU] StreamCompaction::Naive::scan at line 43 took 94.0331 ms
    passed
==== naive scan, non-power-of-two ====
[GPU] StreamCompaction::Naive::scan at line 43 took 92.5403 ms
    passed
==== work-efficient scan, power-of-two ====
[GPU] StreamCompaction::Efficient::scan at line 175 took 42.7819 ms
    passed
==== work-efficient scan, non-power-of-two ====
[GPU] StreamCompaction::Efficient::scan at line 175 took 42.5946 ms
    passed
==== more-efficient scan, power-of-two ====
[GPU] StreamCompaction::MoreEfficient::scan at line 102 took 8.38566 ms
    passed
==== more-efficient scan, non-power-of-two ====
[GPU] StreamCompaction::MoreEfficient::scan at line 102 took 8.28874 ms
    passed
==== thrust scan, power-of-two ====
[GPU] StreamCompaction::Thrust::scan at line 33 took 3.78246 ms
    passed
==== thrust scan, non-power-of-two ====
[GPU] StreamCompaction::Thrust::scan at line 33 took 3.03533 ms
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
==== cpu compact without scan, power-of-two ====
[CPU] StreamCompaction::CPU::compactWithoutScan at line 106 took 126.6 ms
    passed
==== cpu compact without scan, non-power-of-two ====
[CPU] StreamCompaction::CPU::compactWithoutScan at line 106 took 127.101 ms
    passed
==== cpu compact with scan ====
[CPU] StreamCompaction::CPU::scan at line 17 took 165.578 ms
[CPU] StreamCompaction::CPU::compactWithScan at line 125 took 419.615 ms
    passed
==== work-efficient compact, power-of-two ====
padSize: 67108864
[GPU] StreamCompaction::Efficient::compact at line 269 took 46.5992 ms
    passed
==== work-efficient compact, non-power-of-two ====
padSize: 67108864
[GPU] StreamCompaction::Efficient::compact at line 269 took 46.2794 ms
    passed
```

# Performance Analysis

![](/profile/Graph.png)

Above is a graph of varies implementation of scan algorithm. Tunning block size, I found out the optimal block sizes are 128 or 256. Below is my hardware specs, which demostrate the maximum number of blocks that can be fit in all multiprocessor.

![](/profile/HardwareSpecs.png)

## Bottleneck Analysis (More Efficient Algorithm)

![](/profile/NCompute.png)

Above is a code analysis graph of warp stalls. In the graph, it shows the most significant stall reason is Waiting, which is caused by warp divergence. Future optimization could be eliminate warp divergence by recalculating array indices. 

## Thrust Code Analysis

![](/profile/Thrust.png)

Above is a demonstration of thrust assembly code. This code shows thrust makes use of every resources, including registers(maximum usage is 40+, whereas my code 10+), loop unrolling to hide long scoreboard latency. 