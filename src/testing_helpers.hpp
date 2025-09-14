#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>
#include "termcolor.hpp"

template<typename T>
int cmpArrays(int n, T *a, T *b) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void printDesc(const char *desc) {
    printf("==== %s ====\n", desc);
}

template<typename T>
void printCmpResult(int n, T *a, T *b) {
    bool bSucceeded = !cmpArrays(n, a, b);
    if (bSucceeded)
    {
        std::cout << termcolor::green;
    }else
    {
        std::cout << termcolor::red;
    }
    printf("    %s \n",
       !bSucceeded ? "FAIL VALUE" : "passed");
    std::cout << termcolor::reset;
}

template<typename T>
void printCmpLenResult(int n, int expN, T *a, T *b) {
	bool bSucceeded = (n != -1 && n == expN && !cmpArrays(n, a, b));

    if (bSucceeded)
    {
        std::cout << termcolor::green;
    }
    else
    {
        std::cout << termcolor::red;
    }
    if (n != expN) {
        printf("    expected %d elements, got %d\n", expN, n);
    }
    printf("    %s \n",
            (n == -1 || n != expN) ? "FAIL COUNT" :
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
    std::cout << termcolor::reset;
}

void zeroArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}

void onesArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
}

void genArray(int n, int *a, int maxval) {
    srand(time(nullptr));

    for (int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
    }
}

void printArray(int n, int *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

//template<typename T>
//void printElapsedTime(T time, std::string note = "", const char* unit)
//{
//    std::cout <<"   elapsed time: " << termcolor::yellow << time << unit<< "    " << termcolor::reset << note <<  std::endl;
//}

template<typename T>
void printElapsedTime(T timeStruct, std::string note = "")
{
    std::cout << "   elapsed time: " << termcolor::yellow << timeStruct.time << timeStruct.unit << "    " << termcolor::reset << note << std::endl;
}
