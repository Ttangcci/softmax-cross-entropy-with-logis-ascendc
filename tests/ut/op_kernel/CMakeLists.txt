# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    set(softmax_cross_entropy_with_logits_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/softmax_cross_entropy_with_logits_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/softmax_cross_entropy_with_logits_infershape.cpp
    )
    AddOpTestCase(
        softmax_cross_entropy_with_logits
        "ascend910b"
        ""
        "${softmax_cross_entropy_with_logits_tiling_files}")
endif()
