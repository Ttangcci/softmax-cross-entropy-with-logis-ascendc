import argparse
from dataclasses import dataclass

import numpy as np


DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
}


@dataclass
class Case:
    name: str
    features: np.ndarray
    labels: np.ndarray
    dtype: str = "float32"


def softmax_cross_entropy_with_logits(features, labels, out_dtype=np.float32):
    features_fp32 = features.astype(np.float32)
    labels_fp32 = labels.astype(np.float32)
    shifted = features_fp32 - np.max(features_fp32, axis=-1, keepdims=True)
    exp_value = np.exp(shifted)
    softmax = exp_value / np.sum(exp_value, axis=-1, keepdims=True)
    loss = -np.sum(labels_fp32 * (shifted - np.log(np.sum(exp_value, axis=-1, keepdims=True))), axis=-1)
    backprop = softmax - labels_fp32
    return loss.astype(out_dtype), backprop.astype(out_dtype)


def float32_to_bfloat16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bfloat16_to_float32(value):
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def one_hot(labels, depth):
    result = np.zeros(labels.shape + (depth,), dtype=np.float32)
    np.put_along_axis(result, labels[..., None], 1.0, axis=-1)
    return result


def make_probability_labels(rng, shape):
    labels = rng.random(shape, dtype=np.float32)
    return labels / np.sum(labels, axis=-1, keepdims=True)


def make_cases():
    rng = np.random.default_rng(20240617)
    fixed_features = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    fixed_indices = np.array([4, 0, 2, 0])

    cases = [
        Case("fixed_one_hot_f32", fixed_features, one_hot(fixed_indices, fixed_features.shape[-1])),
    ]

    for shape in [(1, 3), (7, 5), (2, 3, 4), (2, 2, 3, 5), (8, 4096), (2, 16384)]:
        features = rng.normal(loc=0.0, scale=2.0, size=shape).astype(np.float32)
        labels = make_probability_labels(rng, shape)
        cases.append(Case(f"prob_f32_shape_{'x'.join(map(str, shape))}", features, labels, "float32"))
        cases.append(Case(f"prob_f16_shape_{'x'.join(map(str, shape))}", features, labels, "float16"))
        cases.append(Case(f"prob_bf16_shape_{'x'.join(map(str, shape))}", features, labels, "bfloat16"))

    return cases


def run_case(case, verbose=False):
    if case.dtype == "bfloat16":
        features = bfloat16_to_float32(float32_to_bfloat16(case.features))
        labels = bfloat16_to_float32(float32_to_bfloat16(case.labels))
        loss_fp32, backprop_fp32 = softmax_cross_entropy_with_logits(features, labels, np.float32)
        loss = bfloat16_to_float32(float32_to_bfloat16(loss_fp32))
        backprop = bfloat16_to_float32(float32_to_bfloat16(backprop_fp32))
    else:
        out_dtype = DTYPE_MAP[case.dtype]
        features = case.features.astype(out_dtype)
        labels = case.labels.astype(out_dtype)
        loss, backprop = softmax_cross_entropy_with_logits(features, labels, out_dtype)
    ref_loss, ref_backprop = softmax_cross_entropy_with_logits(case.features, case.labels, np.float32)

    loss_error = np.max(np.abs(loss.astype(np.float32) - ref_loss))
    backprop_error = np.max(np.abs(backprop.astype(np.float32) - ref_backprop))
    print(
        f"{case.name}: dtype={case.dtype}, features={features.shape}, "
        f"loss={loss.shape}, backprop={backprop.shape}, "
        f"max_loss_err={loss_error:.6g}, max_backprop_err={backprop_error:.6g}"
    )
    if verbose:
        print("loss:", loss)
        print("backprop:", backprop)


def main():
    parser = argparse.ArgumentParser(description="Generate NumPy reference outputs for SoftmaxCrossEntropyWithLogits.")
    parser.add_argument("--case", default="all", help="Case name to run, or all.")
    parser.add_argument("--verbose", action="store_true", help="Print full loss and backprop arrays.")
    args = parser.parse_args()

    cases = make_cases()
    selected = [case for case in cases if args.case == "all" or case.name == args.case]
    if not selected:
        names = ", ".join(case.name for case in cases)
        raise ValueError(f"unknown case {args.case!r}. Available cases: {names}")

    for case in selected:
        run_case(case, args.verbose)


if __name__ == "__main__":
    main()
