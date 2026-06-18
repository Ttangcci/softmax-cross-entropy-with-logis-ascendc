#!/usr/bin/env python3

import sys

import numpy as np


def compare(name, dtype, rtol, atol):
    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
    output = np.fromfile(f"{dtype}_output_{name}.bin", dtype=np_dtype).astype(np.float32)
    golden = np.fromfile(f"{dtype}_golden_{name}.bin", dtype=np_dtype).astype(np.float32)
    if output.shape != golden.shape:
        print(f"{name}: shape mismatch, output={output.shape}, golden={golden.shape}")
        return False
    passed = np.allclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
    max_error = np.max(np.abs(output - golden)) if output.size else 0.0
    print(f"{name}: {'PASSED' if passed else 'FAILED'}, max_abs_error={max_error}")
    return passed


if __name__ == "__main__":
    dtype = sys.argv[1]
    tolerance = 1e-4 if dtype == "float32" else 5e-3
    success = compare("loss", dtype, tolerance, tolerance)
    success = compare("backprop", dtype, tolerance, tolerance) and success
    raise SystemExit(0 if success else 1)
