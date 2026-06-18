/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_cross_entropy_with_logits_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferShapeSoftmaxCrossEntropyWithLogits(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSoftmaxCrossEntropyWithLogits");

    const gert::Shape* featuresShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    const gert::Shape* labelsShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelsShape);

    size_t dimNum = featuresShape->GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "features rank must be at least 2"), return GRAPH_FAILED);
    OP_CHECK_IF(labelsShape->GetDimNum() != dimNum,
                OP_LOGE(context, "features and labels must have the same rank"), return GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; ++i) {
        int64_t featuresDim = featuresShape->GetDim(i);
        int64_t labelsDim = labelsShape->GetDim(i);
        OP_CHECK_IF(featuresDim >= 0 && labelsDim >= 0 && featuresDim != labelsDim,
                    OP_LOGE(context, "features and labels must have the same shape"), return GRAPH_FAILED);
    }

    gert::Shape* lossShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);

    gert::Shape* backpropShape = context->GetOutputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, backpropShape);

    lossShape->SetDimNum(dimNum - 1);
    for (size_t i = 0; i < dimNum - 1; i++) {
        lossShape->SetDim(i, featuresShape->GetDim(i));
    }

    *backpropShape = *featuresShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSoftmaxCrossEntropyWithLogits");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftmaxCrossEntropyWithLogits).InferShape(InferShapeSoftmaxCrossEntropyWithLogits);
} // namespace ops
