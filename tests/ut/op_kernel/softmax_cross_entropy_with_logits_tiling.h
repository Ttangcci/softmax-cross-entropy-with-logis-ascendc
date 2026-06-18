/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TEST_TILING_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TEST_TILING_H

#include <cstdint>
#include <cstring>
#include "../../../op_kernel/softmax_cross_entropy_with_logits_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(
    const __gm__ uint8_t* tiling, SoftmaxCrossEntropyWithLogitsTilingData* tilingData)
{
    const __gm__ uint32_t* src = reinterpret_cast<const __gm__ uint32_t*>(tiling);
    uint32_t* dst = reinterpret_cast<uint32_t*>(tilingData);
    for (size_t i = 0; i < sizeof(SoftmaxCrossEntropyWithLogitsTilingData) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}
#else
inline void InitTilingData(uint8_t* tiling, SoftmaxCrossEntropyWithLogitsTilingData* tilingData)
{
    memcpy(tilingData, tiling, sizeof(SoftmaxCrossEntropyWithLogitsTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif
