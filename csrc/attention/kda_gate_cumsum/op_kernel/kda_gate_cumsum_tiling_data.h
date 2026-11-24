#pragma once

#include <cstdint>

struct KdaGateCumsumKernelTilingData {
    int64_t batch;
    int64_t t;
    int64_t hv;
    int64_t k;
    int64_t rank;
    int64_t layout;
    int64_t chunkSize;
    int64_t seqNum;
    int64_t hasCuSeqlens;
    int64_t hasALog;
    int64_t hasDtBias;
    int64_t dataType;
    int64_t useGateInKernel;
    int64_t safeGate;
    float lowerBound;
    int64_t usedCoreNum;
};
