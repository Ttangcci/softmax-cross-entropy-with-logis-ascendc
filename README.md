# SoftmaxCrossEntropyWithLogits

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品 | √ |

## 功能说明

- 算子功能：根据 logits 和 labels 计算 softmax 交叉熵损失，并输出 logits 对应的反向梯度。

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

其中最后一维为类别维度。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>输入 logits，最后一维为类别维度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labels</td>
      <td>输入</td>
      <td>标签分布，与 features 同 shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>输出</td>
      <td>交叉熵损失，shape 为 features 去除最后一维。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprop</td>
      <td>输出</td>
      <td>对 logits 的梯度，与 features 同 shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- features 与 labels 必须 shape 相同、dtype 相同。
- 输入 rank 至少为 2，最后一维作为类别维。
- 暂不支持广播。
- 暂不支持 int64、double 数据类型。
- FLOAT16、BFLOAT16 输入在 Kernel 内转换为 FLOAT 进行核心计算，输出再转换回原 dtype。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| aclnn调用 | [test_aclnn_softmax_cross_entropy_with_logits.cpp](./examples/test_aclnn_softmax_cross_entropy_with_logits.cpp) | 通过 [aclnnSoftmaxCrossEntropyWithLogits](./docs/aclnnSoftmaxCrossEntropyWithLogits.md) 接口方式调用算子。 |

## 测试说明

可先使用 NumPy 参考脚本生成对拍基准：

```bash
python tests/val.py
python tests/val.py --case fixed_one_hot_f32 --verbose
```

正式 Ascend C 编译、安装和 aclnn 样例运行需要在已安装 CANN Toolkit 且具备 NPU 的环境中进行。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Ttangcci | 个人开发者 | SoftmaxCrossEntropyWithLogits | 2026 | SoftmaxCrossEntropyWithLogits 算子适配开源仓 |
