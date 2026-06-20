/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/softmax_cross_entropy_with_logits_tiling_data.h"

class SoftmaxCrossEntropyWithLogitsTiling : public testing::Test {};

struct SoftmaxCrossEntropyWithLogitsCompileInfo {};

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, float_full_row)
{
    SoftmaxCrossEntropyWithLogitsCompileInfo compileInfo;
    gert::TilingContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        64,
        262144,
        4096);

    uint64_t expectedTilingKey = 0;
    std::string expectedTilingData = "4 5 1 1 1 5 ";
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(
        context, ge::GRAPH_SUCCESS, expectedTilingKey, expectedTilingData, expectedWorkspaces);
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, float_split_r)
{
    SoftmaxCrossEntropyWithLogitsCompileInfo compileInfo;
    gert::TilingContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{2, 16384}, {2, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16384}, {2, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16384}, {2, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        64,
        262144,
        4096);

    uint64_t expectedTilingKey = 0;
    std::string expectedTilingData = "2 16384 1 1 1 9360 ";
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(
        context, ge::GRAPH_SUCCESS, expectedTilingKey, expectedTilingData, expectedWorkspaces);
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, float16_full_row)
{
    SoftmaxCrossEntropyWithLogitsCompileInfo compileInfo;
    gert::TilingContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        64,
        262144,
        4096);

    uint64_t expectedTilingKey = 1;
    std::string expectedTilingData = "8 32 1 1 1 32 ";
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(
        context, ge::GRAPH_SUCCESS, expectedTilingKey, expectedTilingData, expectedWorkspaces);
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, reject_dtype_mismatch)
{
    SoftmaxCrossEntropyWithLogitsCompileInfo compileInfo;
    gert::TilingContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {},
        &compileInfo,
        64,
        262144,
        4096);

    uint64_t expectedTilingKey = 0;
    std::string expectedTilingData;
    std::vector<size_t> expectedWorkspaces;
    ExecuteTestCase(
        context, ge::GRAPH_FAILED, expectedTilingKey, expectedTilingData, expectedWorkspaces);
}
