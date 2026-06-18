/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "../../../op_kernel/softmax_cross_entropy_with_logits.cpp"
#include "softmax_cross_entropy_with_logits_tiling.h"
#include <cstdlib>
#include <string>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {
const std::string ROOT_PATH = "../../../../";
const std::string DATA_SOURCE_PATH =
    ROOT_PATH + "experimental/math/softmax_cross_entropy_with_logits/tests/ut/op_kernel/"
                "softmax_cross_entropy_with_logits_data";
const std::string DATA_PATH = "./softmax_cross_entropy_with_logits_data";

class SoftmaxCrossEntropyWithLogitsKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        const std::string copyCommand = "cp -rf " + DATA_SOURCE_PATH + " ./";
        ASSERT_EQ(system(copyCommand.c_str()), 0);
        ASSERT_EQ(system(("chmod -R 755 " + DATA_PATH).c_str()), 0);
    }
};

void RunFloatCase(const std::string& shape, uint64_t batchSize, uint64_t numClasses, uint64_t classTileLength)
{
    const std::string generateCommand =
        "cd " + DATA_PATH + " && python3 gen_data.py '" + shape + "' float32";
    ASSERT_EQ(system(generateCommand.c_str()), 0);

    size_t inputByteSize = batchSize * numClasses * sizeof(float);
    size_t lossByteSize = batchSize * sizeof(float);
    auto* features = static_cast<uint8_t*>(AscendC::GmAlloc(inputByteSize));
    auto* labels = static_cast<uint8_t*>(AscendC::GmAlloc(inputByteSize));
    auto* loss = static_cast<uint8_t*>(AscendC::GmAlloc(lossByteSize));
    auto* backprop = static_cast<uint8_t*>(AscendC::GmAlloc(inputByteSize));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SoftmaxCrossEntropyWithLogitsTilingData)));

    ReadFile(DATA_PATH + "/float32_features.bin", inputByteSize, features, inputByteSize);
    ReadFile(DATA_PATH + "/float32_labels.bin", inputByteSize, labels, inputByteSize);

    auto* tilingData = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tiling);
    tilingData->batchSize = batchSize;
    tilingData->numClasses = numClasses;
    tilingData->blockLength = batchSize;
    tilingData->tileNum = batchSize;
    tilingData->tileLength = 1;
    tilingData->classTileLength = classTileLength;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        softmax_cross_entropy_with_logits<0>,
        1,
        features,
        labels,
        loss,
        backprop,
        workspace,
        tiling);

    WriteFile(DATA_PATH + "/float32_output_loss.bin", loss, lossByteSize);
    WriteFile(DATA_PATH + "/float32_output_backprop.bin", backprop, inputByteSize);

    AscendC::GmFree(features);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(backprop);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    ASSERT_EQ(system(("cd " + DATA_PATH + " && python3 compare_data.py float32").c_str()), 0);
}
} // namespace

TEST_F(SoftmaxCrossEntropyWithLogitsKernel, float_full_row)
{
    RunFloatCase("(4, 5)", 4, 5, 5);
}

TEST_F(SoftmaxCrossEntropyWithLogitsKernel, float_split_r)
{
    RunFloatCase("(2, 16384)", 2, 16384, 4096);
}
