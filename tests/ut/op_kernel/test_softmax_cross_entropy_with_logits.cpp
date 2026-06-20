/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "../../../op_kernel/softmax_cross_entropy_with_logits.cpp"
#include "softmax_cross_entropy_with_logits_tiling.h"
#include <cstdint>
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

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

class SoftmaxCrossEntropyWithLogitsKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        const std::string copyCommand = "cp -rf " + DATA_SOURCE_PATH + " ./";
        ASSERT_EQ(system(copyCommand.c_str()), 0);
        ASSERT_EQ(system(("chmod -R 755 " + DATA_PATH).c_str()), 0);
    }
};

template <uint32_t SchMode>
void RunCase(
    const std::string& shape,
    const std::string& dtype,
    size_t typeSize,
    uint64_t batchSize,
    uint64_t numClasses,
    uint64_t classTileLength)
{
    const std::string generateCommand =
        "cd " + DATA_PATH + " && python3 gen_data.py '" + shape + "' " + dtype;
    ASSERT_EQ(system(generateCommand.c_str()), 0);

    size_t inputByteSize = batchSize * numClasses * typeSize;
    size_t lossByteSize = batchSize * typeSize;
    auto* features = static_cast<uint8_t*>(AscendC::GmAlloc(AlignUp(inputByteSize, 32)));
    auto* labels = static_cast<uint8_t*>(AscendC::GmAlloc(AlignUp(inputByteSize, 32)));
    auto* loss = static_cast<uint8_t*>(AscendC::GmAlloc(AlignUp(lossByteSize, 32)));
    auto* backprop = static_cast<uint8_t*>(AscendC::GmAlloc(AlignUp(inputByteSize, 32)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SoftmaxCrossEntropyWithLogitsTilingData)));

    ReadFile(DATA_PATH + "/" + dtype + "_features.bin", inputByteSize, features, inputByteSize);
    ReadFile(DATA_PATH + "/" + dtype + "_labels.bin", inputByteSize, labels, inputByteSize);

    auto* tilingData = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tiling);
    tilingData->batchSize = batchSize;
    tilingData->numClasses = numClasses;
    tilingData->blockLength = batchSize;
    tilingData->tileNum = batchSize;
    tilingData->tileLength = 1;
    tilingData->classTileLength = classTileLength;

    ICPU_SET_TILING_KEY(SchMode);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kernel = softmax_cross_entropy_with_logits<SchMode>;
    ICPU_RUN_KF(kernel, 1, features, labels, loss, backprop, workspace, tiling);

    WriteFile(DATA_PATH + "/" + dtype + "_output_loss.bin", loss, lossByteSize);
    WriteFile(DATA_PATH + "/" + dtype + "_output_backprop.bin", backprop, inputByteSize);

    AscendC::GmFree(features);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(backprop);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    ASSERT_EQ(system(("cd " + DATA_PATH + " && python3 compare_data.py " + dtype).c_str()), 0);
}
} // namespace

TEST_F(SoftmaxCrossEntropyWithLogitsKernel, float_full_row)
{
    RunCase<0>("(4, 5)", "float32", sizeof(float), 4, 5, 5);
}

TEST_F(SoftmaxCrossEntropyWithLogitsKernel, float_split_r)
{
    RunCase<0>("(2, 16384)", "float32", sizeof(float), 2, 16384, 4096);
}

TEST_F(SoftmaxCrossEntropyWithLogitsKernel, float16_full_row)
{
    RunCase<1>("(8, 32)", "float16", sizeof(uint16_t), 8, 32, 32);
}
