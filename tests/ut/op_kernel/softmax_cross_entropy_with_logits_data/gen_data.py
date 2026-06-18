#!/usr/bin/env python3

import os
import sys

import numpy as np


def parse_shape(value):
    return tuple(int(item.strip()) for item in value.strip("()").split(",") if item.strip())


def generate(shape, dtype):
    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
    rng = np.random.default_rng(20260618)
    features_fp32 = rng.normal(0.0, 2.0, size=shape).astype(np.float32)
    labels_fp32 = rng.random(shape, dtype=np.float32)
    labels_fp32 /= np.sum(labels_fp32, axis=-1, keepdims=True)

    features = features_fp32.astype(np_dtype)
    labels = labels_fp32.astype(np_dtype)
    calc_features = features.astype(np.float32)
    calc_labels = labels.astype(np.float32)
    shifted = calc_features - np.max(calc_features, axis=-1, keepdims=True)
    exp_value = np.exp(shifted)
    sum_value = np.sum(exp_value, axis=-1, keepdims=True)
    loss = -np.sum(calc_labels * (shifted - np.log(sum_value)), axis=-1)
    backprop = exp_value / sum_value - calc_labels

    features.tofile(f"{dtype}_features.bin")
    labels.tofile(f"{dtype}_labels.bin")
    loss.astype(np_dtype).tofile(f"{dtype}_golden_loss.bin")
    backprop.astype(np_dtype).tofile(f"{dtype}_golden_backprop.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: gen_data.py SHAPE DTYPE")
    for file_name in os.listdir("."):
        if file_name.endswith(".bin"):
            os.remove(file_name)
    generate(parse_shape(sys.argv[1]), sys.argv[2])
