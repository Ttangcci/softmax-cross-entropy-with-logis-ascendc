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
                                uint64_t blockLength, uint64_t tileNum, uint64_t tileLength);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void Compute(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void CopyOut(int32_t rowOffset, int32_t rowCount);

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
    uint64_t blockOffset;
};

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Init(
    GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop,
    uint64_t batchSize, uint64_t numClasses,
    uint64_t blockLength, uint64_t /*tileNum*/, uint64_t tileLength)
{
    this->batchSize   = batchSize;
    this->numClasses  = numClasses;
    this->tileLength  = tileLength;
    this->blockOffset = blockLength * GetBlockIdx();
    uint64_t remainRows = batchSize > blockOffset ? batchSize - blockOffset : 0;
    this->blockLength = remainRows < blockLength ? remainRows : blockLength;
    this->tileNum = this->blockLength / tileLength;

    featuresGm.SetGlobalBuffer((__gm__ T*)features + blockOffset * numClasses, this->blockLength * numClasses);
    labelsGm.SetGlobalBuffer((__gm__ T*)labels + blockOffset * numClasses, this->blockLength * numClasses);
    lossGm.SetGlobalBuffer((__gm__ T*)loss + blockOffset, this->blockLength);
    backpropGm.SetGlobalBuffer((__gm__ T*)backprop + blockOffset * numClasses, this->blockLength * numClasses);

    pipe.InitBuffer(featuresQueue, BUFFER_NUM, tileLength * numClasses * sizeof(T));
    pipe.InitBuffer(labelsQueue,   BUFFER_NUM, tileLength * numClasses * sizeof(T));
    pipe.InitBuffer(lossQueue,     BUFFER_NUM, tileLength * sizeof(T));
    pipe.InitBuffer(backpropQueue, BUFFER_NUM, tileLength * numClasses * sizeof(T));
    pipe.InitBuffer(tmpBuf,        tileLength * numClasses * sizeof(float));
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        pipe.InitBuffer(castBuf, (3 * tileLength * numClasses + tileLength) * sizeof(float));
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
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Process()
{
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
