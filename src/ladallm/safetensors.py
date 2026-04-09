"""Safetensors weight loader for LadaLLM.

Provides memory-mapped loading of HuggingFace safetensors format
for efficient model weight access without copying data.
"""

import json
import mmap
import struct

import ml_dtypes
import numpy as np


class Safetensors:
    """Memory-mapped safetensors loader.

    Loads model weights from safetensors files with zero-copy
    tensor views via memory mapping.
    """

    DTYPE_MAP = {
        "bool": (np.bool_, 1),
        "u8": (np.uint8, 1),
        "i8": (np.int8, 1),
        "i16": (np.int16, 2),
        "u16": (np.uint16, 2),
        "f16": (np.float16, 2),
        "bf16": (ml_dtypes.bfloat16, 2),
        "i32": (np.int32, 4),
        "u32": (np.uint32, 4),
        "f32": (np.float32, 4),
        "i64": (np.int64, 8),
        "u64": (np.uint64, 8),
        "f64": (np.float64, 8),
    }

    def close(self) -> None:
        """Close the memory map and release resources."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __init__(self, path: str):
        """Load safetensors file from path.

        Args:
            path: Path to .safetensors file
        """
        self.header_length: int
        self.header: dict = {}
        self.tensor_data: dict = {}
        self._mmap = None
        with open(path, "rb") as f:
            self.header_length = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(self.header_length).decode("utf-8"))
            data_start = (
                8 + self.header_length + (8 - (8 + self.header_length) % 8) % 8
            )  # 8 + header size + padding

            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            for name, info in self.header.items():
                if name == "__metadata__":
                    continue

                start = data_start + info["data_offsets"][0]
                end = data_start + info["data_offsets"][1]
                shape = info["shape"]
                dtype, size = self.DTYPE_MAP[info["dtype"].lower()]
                self.tensor_data[name] = np.frombuffer(
                    self._mmap, dtype=dtype, offset=start, count=(end - start) // size
                ).reshape(shape)
