/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../op_kernel/softmax_cross_entropy_with_logits_tiling_data.h"
#include "../op_kernel/softmax_cross_entropy_with_logits_tiling_key.h"

namespace optiling {

struct SoftmaxCrossEntropyWithLogitsCompileInfo {};

static ge::graphStatus TilingParseForSoftmaxCrossEntropyWithLogits(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftmaxCrossEntropyWithLogitsTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    SoftmaxCrossEntropyWithLogitsTilingData* tiling =
        context->GetTilingData<SoftmaxCrossEntropyWithLogitsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftmaxCrossEntropyWithLogitsTilingData), 0,
                 sizeof(SoftmaxCrossEntropyWithLogitsTilingData)) != EOK,
        OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum <= 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize == 0"), return ge::GRAPH_FAILED);

    const gert::StorageShape* featuresStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresStorageShape);
    const gert::Shape& featuresShape = featuresStorageShape->GetStorageShape();

    int64_t dimNum = featuresShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "features must be at least 2D"), return ge::GRAPH_FAILED);

    int64_t batchSize = 1;
    for (int64_t i = 0; i < dimNum - 1; i++) {
        batchSize *= featuresShape.GetDim(i);
    }
    int64_t numClasses = featuresShape.GetDim(dimNum - 1);
    OP_CHECK_IF(batchSize <= 0 || numClasses <= 0,
                OP_LOGE(context, "batch size and num classes must be positive"), return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(0)->GetDataType();
    auto labelsDtype = context->GetInputDesc(1)->GetDataType();
    OP_CHECK_IF(labelsDtype != dtype,
                OP_LOGE(context, "features and labels must have the same dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "only float32, float16, and bfloat16 are supported by current kernel"),
                return ge::GRAPH_FAILED);
    int64_t typeLength = (dtype == ge::DT_FLOAT) ? 4 : 2;

    constexpr int64_t bufferNum = 2;
    int64_t queueBytesPerRow = bufferNum * (3 * numClasses * typeLength + typeLength);
    int64_t calcBytesPerRow = numClasses * sizeof(float);
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        calcBytesPerRow += (3 * numClasses + 1) * sizeof(float);
    }
    int64_t bytesPerRow = queueBytesPerRow + calcBytesPerRow + 32;

    int64_t usedCoreNum = batchSize < coreNum ? batchSize : coreNum;
    int64_t blockLength = (batchSize + usedCoreNum - 1) / usedCoreNum;

    int64_t classTileLength = numClasses;
    int64_t tileLength = 1;
    if (bytesPerRow <= static_cast<int64_t>(ubSize)) {
        int64_t maxRowsPerTile = static_cast<int64_t>(ubSize) / bytesPerRow;
        if (maxRowsPerTile <= 0) maxRowsPerTile = 1;
        tileLength = blockLength < maxRowsPerTile ? blockLength : maxRowsPerTile;
    } else {
        int64_t bytesPerClass = bufferNum * 3 * typeLength + sizeof(float);
        if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
            bytesPerClass += 3 * sizeof(float);
        }
        int64_t availableBytes = static_cast<int64_t>(ubSize) - bufferNum * typeLength - 32;
        int64_t maxClassesPerTile = availableBytes / bytesPerClass;
        OP_CHECK_IF(maxClassesPerTile <= 0,
                    OP_LOGE(context, "UB is too small for split class tiling"), return ge::GRAPH_FAILED);
        classTileLength = numClasses < maxClassesPerTile ? numClasses : maxClassesPerTile;
    }
    if (tileLength <= 0) tileLength = 1;
    int64_t tileNum = blockLength / tileLength;

    tiling->batchSize   = static_cast<uint64_t>(batchSize);
    tiling->numClasses  = static_cast<uint64_t>(numClasses);
    tiling->blockLength = static_cast<uint64_t>(blockLength);
    tiling->tileNum     = static_cast<uint64_t>(tileNum);
    tiling->tileLength  = static_cast<uint64_t>(tileLength);
    tiling->classTileLength = static_cast<uint64_t>(classTileLength);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(SCH_MODE_FLOAT);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SCH_MODE_FLOAT16);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SCH_MODE_BF16);
    }
    context->SetTilingKey(tilingKey);
    
    context->SetBlockDim(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SoftmaxCrossEntropyWithLogits)
    .Tiling(SoftmaxCrossEntropyWithLogitsTilingFunc)
    .TilingParse<SoftmaxCrossEntropyWithLogitsCompileInfo>(TilingParseForSoftmaxCrossEntropyWithLogits);

} // namespace optiling
