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
 * \file softmax_cross_entropy_with_logits_proto.h
 * \brief
*/

#ifndef OPS_OP_PROTO_INC_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define OPS_OP_PROTO_INC_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Computes softmax cross entropy loss and backprop.

*@par Inputs:
* @li features: A logits tensor. Must be one of the following types: float16, float32, bfloat16.
* @li labels: A labels tensor with the same shape as features. Must be one of the following types: float16,
* float32, bfloat16.
*@par Outputs:
* @li loss: A tensor with the class dimension reduced. Must be one of the following types: float16, float32,
* bfloat16.
* @li backprop: A tensor with the same shape as features. Must be one of the following types: float16,
* float32, bfloat16.

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SoftmaxCrossEntropyWithLogits.
*/

REG_OP(SoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(labels, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyWithLogits)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
