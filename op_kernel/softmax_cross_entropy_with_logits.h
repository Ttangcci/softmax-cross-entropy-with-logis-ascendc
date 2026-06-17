#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softmax_cross_entropy_with_logits_tiling_data.h"
#include "softmax_cross_entropy_with_logits_tiling_key.h"

namespace NsSoftmaxCrossEntropyWithLogits {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelSoftmaxCrossEntropyWithLogits {
public:
    __aicore__ inline KernelSoftmaxCrossEntropyWithLogits() {};
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop,
                                uint64_t batchSize, uint64_t numClasses,
                                uint64_t blockLength, uint64_t tileNum, uint64_t tileLength,
                                uint64_t classTileLength);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void Compute(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void CopyOut(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void ProcessSplitR();
    __aicore__ inline void CopyInFeaturesClass(int32_t rowOffset, int32_t classOffset, int32_t classCount);
    __aicore__ inline void CopyInFeatureLabelClass(int32_t rowOffset, int32_t classOffset, int32_t classCount);
    __aicore__ inline void CopyOutBackpropClass(int32_t rowOffset, int32_t classOffset, int32_t classCount);
    __aicore__ inline void CopyOutLoss(int32_t rowOffset);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> featuresQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> labelsQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> lossQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> backpropQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> lnBuf;

    GlobalTensor<T> featuresGm;
    GlobalTensor<T> labelsGm;
    GlobalTensor<T> lossGm;
    GlobalTensor<T> backpropGm;

    uint64_t batchSize;
    uint64_t numClasses;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t classTileLength;
    uint64_t blockOffset;
};

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Init(
    GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop,
    uint64_t batchSize, uint64_t numClasses,
    uint64_t blockLength, uint64_t /*tileNum*/, uint64_t tileLength, uint64_t classTileLength)
{
    this->batchSize   = batchSize;
    this->numClasses  = numClasses;
    this->tileLength  = tileLength;
    this->classTileLength = classTileLength;
    this->blockOffset = blockLength * GetBlockIdx();
    uint64_t remainRows = batchSize > blockOffset ? batchSize - blockOffset : 0;
    this->blockLength = remainRows < blockLength ? remainRows : blockLength;
    this->tileNum = this->blockLength / tileLength;

    featuresGm.SetGlobalBuffer((__gm__ T*)features + blockOffset * numClasses, this->blockLength * numClasses);
    labelsGm.SetGlobalBuffer((__gm__ T*)labels + blockOffset * numClasses, this->blockLength * numClasses);
    lossGm.SetGlobalBuffer((__gm__ T*)loss + blockOffset, this->blockLength);
    backpropGm.SetGlobalBuffer((__gm__ T*)backprop + blockOffset * numClasses, this->blockLength * numClasses);

    uint64_t processClassLength = classTileLength < numClasses ? classTileLength : numClasses;
    uint64_t processRowLength = classTileLength < numClasses ? 1 : tileLength;
    pipe.InitBuffer(featuresQueue, BUFFER_NUM, processRowLength * processClassLength * sizeof(T));
    pipe.InitBuffer(labelsQueue,   BUFFER_NUM, processRowLength * processClassLength * sizeof(T));
    pipe.InitBuffer(lossQueue,     BUFFER_NUM, processRowLength * sizeof(T));
    pipe.InitBuffer(backpropQueue, BUFFER_NUM, processRowLength * processClassLength * sizeof(T));
    pipe.InitBuffer(tmpBuf,        processRowLength * processClassLength * sizeof(float));
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        pipe.InitBuffer(castBuf, (3 * processRowLength * processClassLength + processRowLength) * sizeof(float));
    }
    pipe.InitBuffer(lnBuf, 32 * sizeof(float));
}

// template <typename T>
// __aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyIn(int32_t rowOffset, int32_t rowCount)
// {
//     LocalTensor<T> featuresLocal = featuresQueue.AllocTensor<T>();
//     LocalTensor<T> labelsLocal   = labelsQueue.AllocTensor<T>();
//     DataCopy(featuresLocal, featuresGm[rowOffset * numClasses], rowCount * numClasses);
//     DataCopy(labelsLocal,   labelsGm[rowOffset * numClasses],   rowCount * numClasses);
//     featuresQueue.EnQue(featuresLocal);
//     labelsQueue.EnQue(labelsLocal);
// }
template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyIn(int32_t rowOffset, int32_t rowCount)
{
    LocalTensor<T> featuresLocal = featuresQueue.AllocTensor<T>();
    LocalTensor<T> labelsLocal   = labelsQueue.AllocTensor<T>();
    
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = rowCount * numClasses * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    
    DataCopyPad(featuresLocal, featuresGm[rowOffset * numClasses], copyParams, padParams);
    DataCopyPad(labelsLocal,   labelsGm[rowOffset * numClasses],   copyParams, padParams);
    
    featuresQueue.EnQue(featuresLocal);
    labelsQueue.EnQue(labelsLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Compute(int32_t rowOffset, int32_t rowCount)
{
    LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
    LocalTensor<T> labelsLocal   = labelsQueue.DeQue<T>();
    LocalTensor<T> backpropLocal = backpropQueue.AllocTensor<T>();
    LocalTensor<T> lossLocal     = lossQueue.AllocTensor<T>();
    LocalTensor<float> tmpLocal  = tmpBuf.Get<float>();

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        int32_t elementCount = rowCount * numClasses;
        LocalTensor<float> featuresFp32 = castBuf.Get<float>();
        LocalTensor<float> labelsFp32 = featuresFp32[tileLength * numClasses];
        LocalTensor<float> backpropFp32 = labelsFp32[tileLength * numClasses];
        LocalTensor<float> lossFp32 = backpropFp32[tileLength * numClasses];

        Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, elementCount);
        Cast(labelsFp32, labelsLocal, RoundMode::CAST_NONE, elementCount);

        for (int32_t i = 0; i < rowCount; i++) {
            int32_t rowStart = i * numClasses;

            float maxVal = featuresFp32.GetValue(rowStart);
            for (int32_t j = 1; j < (int32_t)numClasses; j++) {
                float v = featuresFp32.GetValue(rowStart + j);
                if (v > maxVal) {
                    maxVal = v;
                }
            }

            for (int32_t j = 0; j < (int32_t)numClasses; j++) {
                tmpLocal.SetValue(rowStart + j, featuresFp32.GetValue(rowStart + j) - maxVal);
            }
            Exp(tmpLocal[rowStart], tmpLocal[rowStart], (int32_t)numClasses);

            float sumVal = 0.0f;
            for (int32_t j = 0; j < (int32_t)numClasses; j++) {
                sumVal = sumVal + tmpLocal.GetValue(rowStart + j);
            }

            LocalTensor<float> lnTensor = lnBuf.Get<float>();
            lnTensor.SetValue(0, sumVal);
            Ln(lnTensor, lnTensor, 1);
            float logSumVal = lnTensor.GetValue(0);

            float lossVal = 0.0f;
            for (int32_t j = 0; j < (int32_t)numClasses; j++) {
                float softmax = tmpLocal.GetValue(rowStart + j) / sumVal;
                float label = labelsFp32.GetValue(rowStart + j);
                float feature = featuresFp32.GetValue(rowStart + j);
                backpropFp32.SetValue(rowStart + j, softmax - label);
                lossVal = lossVal - label * (feature - maxVal - logSumVal);
            }
            lossFp32.SetValue(i, lossVal);
        }

        Cast(backpropLocal, backpropFp32, RoundMode::CAST_RINT, elementCount);
        Cast(lossLocal, lossFp32, RoundMode::CAST_RINT, rowCount);
    } else {
        for (int32_t i = 0; i < rowCount; i++) {
            int32_t rowStart = i * numClasses;

            // find max
            T maxVal = featuresLocal.GetValue(rowStart);
            for (int32_t j = 1; j < (int32_t)numClasses; j++) {
                T v = featuresLocal.GetValue(rowStart + j);
                if (v > maxVal) maxVal = v;
            }

            // x - max
            for (int32_t j = 0; j < (int32_t)numClasses; j++) {
                tmpLocal.SetValue(rowStart + j, featuresLocal.GetValue(rowStart + j) - maxVal);
            }

            // exp(x - max)
            Exp(tmpLocal[rowStart], tmpLocal[rowStart], (int32_t)numClasses);

            // sum
            T sumVal = static_cast<T>(0);
            for (int32_t j = 0; j < (int32_t)numClasses; j++) {
                sumVal = sumVal + tmpLocal.GetValue(rowStart + j);
            }

            LocalTensor<float> lnTensor = lnBuf.Get<float>();
            lnTensor.SetValue(0, (float)sumVal);
            Ln(lnTensor, lnTensor, 1);
            T logSumVal = (T)lnTensor.GetValue(0);

            // backprop and loss
            T lossVal = static_cast<T>(0);
            for (int32_t j = 0; j < (int32_t)numClasses; j++) {
                T softmax_j = tmpLocal.GetValue(rowStart + j) / sumVal;
                T label_j   = labelsLocal.GetValue(rowStart + j);
                T feat_j    = featuresLocal.GetValue(rowStart + j);
                backpropLocal.SetValue(rowStart + j, softmax_j - label_j);
                lossVal = lossVal + (label_j * static_cast<T>(-1)) * (feat_j - maxVal - logSumVal);
            }
            lossLocal.SetValue(i, lossVal);
        }
    }

    backpropQueue.EnQue<T>(backpropLocal);
    lossQueue.EnQue<T>(lossLocal);
    featuresQueue.FreeTensor(featuresLocal);
    labelsQueue.FreeTensor(labelsLocal);
}

// template <typename T>
// __aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyOut(int32_t rowOffset, int32_t rowCount)
// {
//     LocalTensor<T> backpropLocal = backpropQueue.DeQue<T>();
//     LocalTensor<T> lossLocal     = lossQueue.DeQue<T>();
//     DataCopy(backpropGm[rowOffset * numClasses], backpropLocal, rowCount * numClasses);
//     DataCopy(lossGm[rowOffset], lossLocal, rowCount);
//     backpropQueue.FreeTensor(backpropLocal);
//     lossQueue.FreeTensor(lossLocal);
// }
template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyOut(int32_t rowOffset, int32_t rowCount)
{
    LocalTensor<T> backpropLocal = backpropQueue.DeQue<T>();
    LocalTensor<T> lossLocal     = lossQueue.DeQue<T>();
    
    DataCopyExtParams copyParamsBackprop;
    copyParamsBackprop.blockCount = 1;
    copyParamsBackprop.blockLen = rowCount * numClasses * sizeof(T);
    copyParamsBackprop.srcStride = 0;
    copyParamsBackprop.dstStride = 0;
    DataCopyPad(backpropGm[rowOffset * numClasses], backpropLocal, copyParamsBackprop);
    
    DataCopyExtParams copyParamsLoss;
    copyParamsLoss.blockCount = 1;
    copyParamsLoss.blockLen = rowCount * sizeof(T);
    copyParamsLoss.srcStride = 0;
    copyParamsLoss.dstStride = 0;
    DataCopyPad(lossGm[rowOffset], lossLocal, copyParamsLoss);
    
    backpropQueue.FreeTensor(backpropLocal);
    lossQueue.FreeTensor(lossLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyInFeaturesClass(
    int32_t rowOffset, int32_t classOffset, int32_t classCount)
{
    LocalTensor<T> featuresLocal = featuresQueue.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = classCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    DataCopyPad(featuresLocal, featuresGm[rowOffset * numClasses + classOffset], copyParams, padParams);
    featuresQueue.EnQue(featuresLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyInFeatureLabelClass(
    int32_t rowOffset, int32_t classOffset, int32_t classCount)
{
    LocalTensor<T> featuresLocal = featuresQueue.AllocTensor<T>();
    LocalTensor<T> labelsLocal = labelsQueue.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = classCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    uint64_t gmOffset = rowOffset * numClasses + classOffset;
    DataCopyPad(featuresLocal, featuresGm[gmOffset], copyParams, padParams);
    DataCopyPad(labelsLocal, labelsGm[gmOffset], copyParams, padParams);

    featuresQueue.EnQue(featuresLocal);
    labelsQueue.EnQue(labelsLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyOutBackpropClass(
    int32_t rowOffset, int32_t classOffset, int32_t classCount)
{
    LocalTensor<T> backpropLocal = backpropQueue.DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = classCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(backpropGm[rowOffset * numClasses + classOffset], backpropLocal, copyParams);

    backpropQueue.FreeTensor(backpropLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyOutLoss(int32_t rowOffset)
{
    LocalTensor<T> lossLocal = lossQueue.DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(lossGm[rowOffset], lossLocal, copyParams);

    lossQueue.FreeTensor(lossLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::ProcessSplitR()
{
    int32_t classTileLen = (int32_t)classTileLength;
    int32_t totalClasses = (int32_t)numClasses;
    int32_t classTileNum = totalClasses / classTileLen;
    int32_t classTail = totalClasses - classTileNum * classTileLen;

    for (int32_t row = 0; row < (int32_t)blockLength; row++) {
        float maxVal = -3.4028234663852886e+38F;

        for (int32_t tile = 0; tile < classTileNum; tile++) {
            int32_t classOffset = tile * classTileLen;
            CopyInFeaturesClass(row, classOffset, classTileLen);
            LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> featuresFp32 = castBuf.Get<float>();
                Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, classTileLen);
                for (int32_t i = 0; i < classTileLen; i++) {
                    float v = featuresFp32.GetValue(i);
                    if (v > maxVal) {
                        maxVal = v;
                    }
                }
            } else {
                for (int32_t i = 0; i < classTileLen; i++) {
                    float v = (float)featuresLocal.GetValue(i);
                    if (v > maxVal) {
                        maxVal = v;
                    }
                }
            }
            featuresQueue.FreeTensor(featuresLocal);
        }
        if (classTail > 0) {
            int32_t classOffset = classTileNum * classTileLen;
            CopyInFeaturesClass(row, classOffset, classTail);
            LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> featuresFp32 = castBuf.Get<float>();
                Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, classTail);
                for (int32_t i = 0; i < classTail; i++) {
                    float v = featuresFp32.GetValue(i);
                    if (v > maxVal) {
                        maxVal = v;
                    }
                }
            } else {
                for (int32_t i = 0; i < classTail; i++) {
                    float v = (float)featuresLocal.GetValue(i);
                    if (v > maxVal) {
                        maxVal = v;
                    }
                }
            }
            featuresQueue.FreeTensor(featuresLocal);
        }

        float sumVal = 0.0F;
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        for (int32_t tile = 0; tile < classTileNum; tile++) {
            int32_t classOffset = tile * classTileLen;
            CopyInFeaturesClass(row, classOffset, classTileLen);
            LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> featuresFp32 = castBuf.Get<float>();
                Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, classTileLen);
                for (int32_t i = 0; i < classTileLen; i++) {
                    tmpLocal.SetValue(i, featuresFp32.GetValue(i) - maxVal);
                }
            } else {
                for (int32_t i = 0; i < classTileLen; i++) {
                    tmpLocal.SetValue(i, (float)featuresLocal.GetValue(i) - maxVal);
                }
            }
            Exp(tmpLocal, tmpLocal, classTileLen);
            for (int32_t i = 0; i < classTileLen; i++) {
                sumVal = sumVal + tmpLocal.GetValue(i);
            }
            featuresQueue.FreeTensor(featuresLocal);
        }
        if (classTail > 0) {
            int32_t classOffset = classTileNum * classTileLen;
            CopyInFeaturesClass(row, classOffset, classTail);
            LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> featuresFp32 = castBuf.Get<float>();
                Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, classTail);
                for (int32_t i = 0; i < classTail; i++) {
                    tmpLocal.SetValue(i, featuresFp32.GetValue(i) - maxVal);
                }
            } else {
                for (int32_t i = 0; i < classTail; i++) {
                    tmpLocal.SetValue(i, (float)featuresLocal.GetValue(i) - maxVal);
                }
            }
            Exp(tmpLocal, tmpLocal, classTail);
            for (int32_t i = 0; i < classTail; i++) {
                sumVal = sumVal + tmpLocal.GetValue(i);
            }
            featuresQueue.FreeTensor(featuresLocal);
        }

        LocalTensor<float> lnTensor = lnBuf.Get<float>();
        lnTensor.SetValue(0, sumVal);
        Ln(lnTensor, lnTensor, 1);
        float logSumVal = lnTensor.GetValue(0);
        float lossVal = 0.0F;

        for (int32_t tile = 0; tile < classTileNum; tile++) {
            int32_t classOffset = tile * classTileLen;
            CopyInFeatureLabelClass(row, classOffset, classTileLen);
            LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
            LocalTensor<T> labelsLocal = labelsQueue.DeQue<T>();
            LocalTensor<T> backpropLocal = backpropQueue.AllocTensor<T>();
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> featuresFp32 = castBuf.Get<float>();
                LocalTensor<float> labelsFp32 = featuresFp32[classTileLen];
                LocalTensor<float> backpropFp32 = labelsFp32[classTileLen];
                Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, classTileLen);
                Cast(labelsFp32, labelsLocal, RoundMode::CAST_NONE, classTileLen);
                for (int32_t i = 0; i < classTileLen; i++) {
                    tmpLocal.SetValue(i, featuresFp32.GetValue(i) - maxVal);
                }
                Exp(tmpLocal, tmpLocal, classTileLen);
                for (int32_t i = 0; i < classTileLen; i++) {
                    float softmax = tmpLocal.GetValue(i) / sumVal;
                    float label = labelsFp32.GetValue(i);
                    float feature = featuresFp32.GetValue(i);
                    backpropFp32.SetValue(i, softmax - label);
                    lossVal = lossVal - label * (feature - maxVal - logSumVal);
                }
                Cast(backpropLocal, backpropFp32, RoundMode::CAST_RINT, classTileLen);
            } else {
                for (int32_t i = 0; i < classTileLen; i++) {
                    tmpLocal.SetValue(i, (float)featuresLocal.GetValue(i) - maxVal);
                }
                Exp(tmpLocal, tmpLocal, classTileLen);
                for (int32_t i = 0; i < classTileLen; i++) {
                    float softmax = tmpLocal.GetValue(i) / sumVal;
                    float label = (float)labelsLocal.GetValue(i);
                    float feature = (float)featuresLocal.GetValue(i);
                    backpropLocal.SetValue(i, (T)(softmax - label));
                    lossVal = lossVal - label * (feature - maxVal - logSumVal);
                }
            }
            backpropQueue.EnQue<T>(backpropLocal);
            CopyOutBackpropClass(row, classOffset, classTileLen);
            featuresQueue.FreeTensor(featuresLocal);
            labelsQueue.FreeTensor(labelsLocal);
        }
        if (classTail > 0) {
            int32_t classOffset = classTileNum * classTileLen;
            CopyInFeatureLabelClass(row, classOffset, classTail);
            LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
            LocalTensor<T> labelsLocal = labelsQueue.DeQue<T>();
            LocalTensor<T> backpropLocal = backpropQueue.AllocTensor<T>();
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> featuresFp32 = castBuf.Get<float>();
                LocalTensor<float> labelsFp32 = featuresFp32[classTileLen];
                LocalTensor<float> backpropFp32 = labelsFp32[classTileLen];
                Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, classTail);
                Cast(labelsFp32, labelsLocal, RoundMode::CAST_NONE, classTail);
                for (int32_t i = 0; i < classTail; i++) {
                    tmpLocal.SetValue(i, featuresFp32.GetValue(i) - maxVal);
                }
                Exp(tmpLocal, tmpLocal, classTail);
                for (int32_t i = 0; i < classTail; i++) {
                    float softmax = tmpLocal.GetValue(i) / sumVal;
                    float label = labelsFp32.GetValue(i);
                    float feature = featuresFp32.GetValue(i);
                    backpropFp32.SetValue(i, softmax - label);
                    lossVal = lossVal - label * (feature - maxVal - logSumVal);
                }
                Cast(backpropLocal, backpropFp32, RoundMode::CAST_RINT, classTail);
            } else {
                for (int32_t i = 0; i < classTail; i++) {
                    tmpLocal.SetValue(i, (float)featuresLocal.GetValue(i) - maxVal);
                }
                Exp(tmpLocal, tmpLocal, classTail);
                for (int32_t i = 0; i < classTail; i++) {
                    float softmax = tmpLocal.GetValue(i) / sumVal;
                    float label = (float)labelsLocal.GetValue(i);
                    float feature = (float)featuresLocal.GetValue(i);
                    backpropLocal.SetValue(i, (T)(softmax - label));
                    lossVal = lossVal - label * (feature - maxVal - logSumVal);
                }
            }
            backpropQueue.EnQue<T>(backpropLocal);
            CopyOutBackpropClass(row, classOffset, classTail);
            featuresQueue.FreeTensor(featuresLocal);
            labelsQueue.FreeTensor(labelsLocal);
        }

        LocalTensor<T> lossLocal = lossQueue.AllocTensor<T>();
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> lossFp32 = castBuf.Get<float>();
            lossFp32.SetValue(0, lossVal);
            Cast(lossLocal, lossFp32, RoundMode::CAST_RINT, 1);
        } else {
            lossLocal.SetValue(0, (T)lossVal);
        }
        lossQueue.EnQue<T>(lossLocal);
        CopyOutLoss(row);
    }
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Process()
{
    if (classTileLength < numClasses) {
        ProcessSplitR();
        return;
    }
    for (int32_t i = 0; i < (int32_t)tileNum; i++) {
        int32_t rowOffset = i * tileLength;
        CopyIn(rowOffset, tileLength);
        Compute(rowOffset, tileLength);
        CopyOut(rowOffset, tileLength);
    }
    int32_t tail = blockLength - tileNum * tileLength;
    if (tail > 0) {
        int32_t rowOffset = tileNum * tileLength;
        CopyIn(rowOffset, tail);
        Compute(rowOffset, tail);
        CopyOut(rowOffset, tail);
    }
}

} // namespace NsSoftmaxCrossEntropyWithLogits
#endif
