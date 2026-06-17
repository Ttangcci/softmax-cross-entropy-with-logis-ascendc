# aclnnSoftmaxCrossEntropyWithLogits

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :----: |
| Atlas A2 训练系列产品 | √ |

## 功能说明

- 算子功能：根据输入 logits 和 labels，沿最后一维计算 softmax 交叉熵损失，并输出对 logits 的梯度。

- 计算公式：

  $$
  p_{i,c} = \frac{\exp(x_{i,c} - \max_c x_{i,c})}{\sum_c \exp(x_{i,c} - \max_c x_{i,c})}
  $$

  $$
  loss_i = -\sum_c y_{i,c}\log(p_{i,c})
  $$

  $$
  backprop_{i,c} = p_{i,c} - y_{i,c}
  $$

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnSoftmaxCrossEntropyWithLogits` 接口执行计算。

```cpp
aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(
    const aclTensor* features,
    aclTensor* labels,
    aclTensor* loss,
    aclTensor* backprop,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnSoftmaxCrossEntropyWithLogits(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream);
```

## aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize

- **参数说明：**

  - features(aclTensor\*, 计算输入)：公式中的 `x`，表示 logits。数据类型支持 FLOAT、FLOAT16、BFLOAT16，数据格式支持 ND，输入 rank 至少为 2，最后一维为类别维。
  - labels(aclTensor\*, 计算输入)：公式中的 `y`，表示标签分布。数据类型、shape、format 需要与 features 一致。
  - loss(aclTensor\*, 计算输出)：交叉熵损失。数据类型与 features 一致，shape 为 features 去除最后一维后的 shape。
  - backprop(aclTensor\*, 计算输出)：logits 梯度。数据类型与 features 一致，shape 与 features 一致。
  - workspaceSize(uint64_t\*, 出参)：返回需要在 Device 侧申请的 workspace 大小。
  - executor(aclOpExecutor\*\*, 出参)：返回 op 执行器，包含算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

```text
第一段接口完成入参校验，出现以下场景时报错：
1. features、labels、loss、backprop、workspaceSize 或 executor 为空指针。
2. features 和 labels 的 dtype 不一致，或 dtype 不在支持范围内。
3. features 和 labels 的 shape 不一致。
4. features 的 rank 小于 2。
5. 输出 loss 或 backprop 的 dtype、shape 不满足算子约束。
```

## aclnnSoftmaxCrossEntropyWithLogits

- **参数说明：**

  - workspace(void\*, 入参)：在 Device 侧申请的 workspace 内存地址。
  - workspaceSize(uint64_t, 入参)：workspace 大小，由第一段接口获取。
  - executor(aclOpExecutor\*, 入参)：op 执行器，包含算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的 Stream。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束说明

- features 和 labels 必须同 shape、同 dtype。
- 暂不支持广播。
- 暂不支持 int64、double 数据类型。
- 当前实现按 ND 最后一维作为类别维完成归约。

## 调用示例

示例代码参见：

```text
examples/test_aclnn_softmax_cross_entropy_with_logits.cpp
```

可先通过 NumPy 参考脚本生成对拍基准：

```bash
python tests/val.py
```
