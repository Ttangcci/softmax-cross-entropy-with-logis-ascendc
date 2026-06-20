/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SoftmaxCrossEntropyWithLogitsInfershape : public testing::Test {};

TEST_F(SoftmaxCrossEntropyWithLogitsInfershape, infershape_2d)
{
    gert::InfershapeContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expected = {
        {4},
        {4, 5},
    };
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expected);
}

TEST_F(SoftmaxCrossEntropyWithLogitsInfershape, infershape_dynamic_multidimensional)
{
    gert::InfershapeContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{2, -1, 16}, {2, -1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, -1, 16}, {2, -1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expected = {
        {2, -1},
        {2, -1, 16},
    };
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expected);
}

TEST_F(SoftmaxCrossEntropyWithLogitsInfershape, reject_shape_mismatch)
{
    gert::InfershapeContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(context, ge::GRAPH_FAILED, expected);
}

TEST_F(SoftmaxCrossEntropyWithLogitsInfershape, reject_rank_less_than_two)
{
    gert::InfershapeContextPara context(
        "SoftmaxCrossEntropyWithLogits",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(context, ge::GRAPH_FAILED, expected);
}
