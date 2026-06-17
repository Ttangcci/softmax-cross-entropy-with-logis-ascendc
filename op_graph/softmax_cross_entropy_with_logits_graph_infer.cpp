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
 * \file softmax_cross_entropy_with_logits_graph_infer.cpp
 * \brief softmax_cross_entropy_with_logits operator graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferDataTypeSoftmaxCrossEntropyWithLogits(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeSoftmaxCrossEntropyWithLogits");

    ge::DataType featuresDtype = context->GetInputDataType(IDX_0);
    ge::DataType labelsDtype = context->GetInputDataType(IDX_1);
    OP_CHECK_IF(featuresDtype != ge::DT_FLOAT16 && featuresDtype != ge::DT_FLOAT && featuresDtype != ge::DT_BF16,
                OP_LOGE(context, "features dtype must be float16, float32, or bfloat16"), return GRAPH_FAILED);
    OP_CHECK_IF(labelsDtype != featuresDtype,
                OP_LOGE(context, "features and labels must have the same dtype"), return GRAPH_FAILED);

    context->SetOutputDataType(IDX_0, featuresDtype);
    context->SetOutputDataType(IDX_1, featuresDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeSoftmaxCrossEntropyWithLogits");
    return GRAPH_SUCCESS;
}

IMPL_OP(SoftmaxCrossEntropyWithLogits).InferDataType(InferDataTypeSoftmaxCrossEntropyWithLogits);

} // namespace ops
