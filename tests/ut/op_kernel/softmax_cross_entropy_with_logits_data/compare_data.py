#!/usr/bin/env python3

import sys

import numpy as np


def bfloat16_to_float32(value):
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def read_data(path, dtype):
    if dtype == "bfloat16":
        return bfloat16_to_float32(np.fromfile(path, dtype=np.uint16))
    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
    return np.fromfile(path, dtype=np_dtype).astype(np.float32)


def compare(name, dtype, rtol, atol):
    output = read_data(f"{dtype}_output_{name}.bin", dtype)
    golden = read_data(f"{dtype}_golden_{name}.bin", dtype)
    if output.shape != golden.shape:
        print(f"{name}: shape mismatch, output={output.shape}, golden={golden.shape}")
        return False
    passed = np.allclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
    max_error = np.max(np.abs(output - golden)) if output.size else 0.0
    print(f"{name}: {'PASSED' if passed else 'FAILED'}, max_abs_error={max_error}")
    return passed


if __name__ == "__main__":
    dtype = sys.argv[1]
    tolerance = {"float32": 1e-4, "float16": 5e-3, "bfloat16": 5e-2}[dtype]
    success = compare("loss", dtype, tolerance, tolerance)
    success = compare("backprop", dtype, tolerance, tolerance) and success
    raise SystemExit(0 if success else 1)
