/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"


const int ARRAY_SIZE = 524288; // feel free to change the ARRAY_SIZE of array
const int NPOT = ARRAY_SIZE - 3; // Non-Power-Of-Two
int *a = new int[ARRAY_SIZE];
int *b = new int[ARRAY_SIZE];
int *c = new int[ARRAY_SIZE];

int testMain() {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(ARRAY_SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[ARRAY_SIZE - 1] = 0;
    //printArray(ARRAY_SIZE, a, false);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(ARRAY_SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(ARRAY_SIZE, b, a);
    //printArray(ARRAY_SIZE, b, false);
    //zeroArray(ARRAY_SIZE, b);
    //printDesc("cpu scan with openmp, power-of-two");
    //StreamCompaction::CPU::scanOMP(ARRAY_SIZE, b, a);
    //printArray(ARRAY_SIZE, b, false);

    zeroArray(ARRAY_SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

 //   zeroArray(ARRAY_SIZE, c);
 //   printDesc("gpu radix sort");
 //   Radix::radix(ARRAY_SIZE, c, a);
	//int* sortedA = new int[ARRAY_SIZE];
 //   memcpy_s(sortedA, ARRAY_SIZE * sizeof(int), a, ARRAY_SIZE * sizeof(int));
	//std::sort(sortedA, sortedA + ARRAY_SIZE);
 //   printCmpResult(ARRAY_SIZE, sortedA, c);
 //   //printArray(ARRAY_SIZE, c, false);

    zeroArray(ARRAY_SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(ARRAY_SIZE, c, a);
    //printArray(ARRAY_SIZE, c, false);
    printCmpResult(ARRAY_SIZE, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(ARRAY_SIZE, c, a);
    //printArray(ARRAY_SIZE, c, true);
    //printArray(ARRAY_SIZE, c, false);
    printCmpResult(ARRAY_SIZE, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    //printArray(ARRAY_SIZE, c, false);
    printCmpResult(NPOT, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("more-efficient scan, power-of-two");
    StreamCompaction::MoreEfficient::scan(ARRAY_SIZE, c, a);
    //printArray(ARRAY_SIZE, c, false);
    printCmpResult(ARRAY_SIZE, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("more-efficient scan, non-power-of-two");
    StreamCompaction::MoreEfficient::scan(NPOT, c, a);
    //printArray(ARRAY_SIZE, c, false);
    printCmpResult(NPOT, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(ARRAY_SIZE, c, a);
    //printArray(ARRAY_SIZE, c, false);
    printCmpResult(ARRAY_SIZE, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    //printArray(NPOT, c, false);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    onesArray(ARRAY_SIZE - 1, a);  // Leave a 0 at the end to test that edge case
    a[ARRAY_SIZE - 1] = 0;
    //printArray(ARRAY_SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(ARRAY_SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(ARRAY_SIZE, b, a);
    expectedCount = count;
    //printArray(count, b, false);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(ARRAY_SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(ARRAY_SIZE, c, a);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(ARRAY_SIZE, c, a);
    //printArray(count, c, false);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(ARRAY_SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    //printArray(count, c, false);
    printCmpLenResult(count, expectedNPOT, b, c);

    //system("pause"); // stop Win32 console from closing on exit
	return 0;
}


int main()
{
 testMain();
 testMain();
 delete[] a;
 delete[] b;
 delete[] c;
 return 0;
}