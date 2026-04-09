"""Test configuration and fixtures for LadaLLM."""

import tempfile

import numpy as np
import pytest


@pytest.fixture
def temp_safetensors_file():
    """Create a temporary safetensors file for testing."""
    import json
    import os
    import struct

    def _create(tensors, dtype_override=None):
        header = {}
        data_bytes = b""
        current_offset = 0

        dtype_map = {
            np.float16: "F16",
            np.float32: "F32",
            np.float64: "F64",
            np.int32: "I32",
            np.int64: "I64",
            np.uint32: "U32",
            np.uint64: "U64",
            np.bool_: "BOOL",
        }

        try:
            import ml_dtypes
            dtype_map[ml_dtypes.bfloat16] = "BF16"
        except ImportError:
            pass

        for name, arr in tensors.items():
            if dtype_override and name in dtype_override:
                dtype_str = dtype_override[name]
            else:
                dtype_str = dtype_map.get(arr.dtype.type, "F32")

            arr_bytes = arr.tobytes()
            size = len(arr_bytes)

            header[name] = {
                "dtype": dtype_str,
                "shape": list(arr.shape),
                "data_offsets": [current_offset, current_offset + size],
            }

            data_bytes += arr_bytes
            current_offset += size

        header_json = json.dumps(header, separators=(",", ":"))
        header_bytes = header_json.encode("utf-8")
        header_len = len(header_bytes)

        data_start = 8 + header_len
        padding = (8 - (data_start % 8)) % 8
        data_start += padding

        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(b"\x00" * padding)
            f.write(data_bytes)
            return f.name

        yield _create
