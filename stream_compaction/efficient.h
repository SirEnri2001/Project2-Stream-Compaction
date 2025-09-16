#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace MoreEfficient {
        void scan(int n, int* odata, const int* idata);

        void scanDevice(int n, int* device_odata, int* device_idata, int recursive_depth = 0);
        int compact(int n, int* odata, const int* idata);
    }
    namespace Efficient {
        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
