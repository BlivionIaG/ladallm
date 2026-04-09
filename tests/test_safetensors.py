"""Unit tests for F1: Weight Loading (Safetensors)."""

import json
import os
import struct
import tempfile

import ml_dtypes
import numpy as np
import pytest

from ladallm.safetensors import Safetensors


class TestSafetensorsDtypeMap:
    """Test dtype mapping completeness."""

    def test_all_common_dtypes_present(self):
        """Verify all expected dtypes are in the map."""
        expected = {
            "bool", "u8", "i8", "i16", "u16", "f16", "bf16",
            "i32", "u32", "f32", "i64", "u64", "f64",
        }
        actual = set(Safetensors.DTYPE_MAP.keys())
        assert expected <= actual, f"Missing dtypes: {expected - actual}"

    def test_dtype_size_correctness(self):
        """Verify byte sizes match numpy/ml_dtypes."""
        for dtype_str, (np_dtype, size) in Safetensors.DTYPE_MAP.items():
            if hasattr(np_dtype, 'nbytes'):
                assert np_dtype(0).nbytes == size, f"{dtype_str} size mismatch"

    def test_bf16_uses_ml_dtypes(self):
        """BF16 should use ml_dtypes.bfloat16."""
        dtype, size = Safetensors.DTYPE_MAP["bf16"]
        assert dtype == ml_dtypes.bfloat16
        assert size == 2


class TestSafetensorsLoading:
    """Test safetensors file loading."""

    def create_test_file(
        self,
        tensors: dict[str, np.ndarray],
        dtype_override: dict[str, str] | None = None,
    ) -> str:
        """Create a temporary safetensors file with given tensors."""
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
            ml_dtypes.bfloat16: "BF16",
        }

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

        # Build file
        header_json = json.dumps(header, separators=(",", ":"))
        header_bytes = header_json.encode("utf-8")
        header_len = len(header_bytes)

        # Padding to 8-byte boundary
        data_start = 8 + header_len
        padding = (8 - (data_start % 8)) % 8
        data_start += padding

        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            # Header length (8 bytes, little-endian)
            f.write(struct.pack("<Q", header_len))
            # Header
            f.write(header_bytes)
            # Padding
            f.write(b"\x00" * padding)
            # Data
            f.write(data_bytes)
            return f.name

    def test_load_single_tensor(self):
        """Load a single F16 tensor."""
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        path = self.create_test_file({"test": expected})

        try:
            st = Safetensors(path)
            assert "test" in st.tensor_data
            actual = st.tensor_data["test"]
            assert actual.shape == (2, 2)
            assert actual.dtype == np.float16
            np.testing.assert_array_almost_equal(actual, expected)
            st.close()
        finally:
            os.unlink(path)

    def test_load_multiple_tensors(self):
        """Load multiple tensors with different shapes."""
        tensors = {
            "embedding": np.random.randn(100, 64).astype(np.float32),
            "layer1.weight": np.random.randn(64, 128).astype(np.float16),
            "layer1.bias": np.zeros(128, dtype=np.float32),
        }
        path = self.create_test_file(tensors)

        try:
            st = Safetensors(path)
            assert len(st.tensor_data) == 3
            assert st.tensor_data["embedding"].shape == (100, 64)
            assert st.tensor_data["layer1.weight"].shape == (64, 128)
            assert st.tensor_data["layer1.bias"].shape == (128,)
            st.close()
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("dtype_str,np_dtype", [
        ("f16", np.float16),
        ("f32", np.float32),
        ("f64", np.float64),
        ("i32", np.int32),
        ("i64", np.int64),
        ("bool", np.bool_),
    ])
    def test_load_different_dtypes(self, dtype_str, np_dtype):
        """Test loading tensors of various dtypes."""
        expected = np.array([1, 2, 3, 4], dtype=np_dtype).reshape(2, 2)
        path = self.create_test_file(
            {"test": expected},
            dtype_override={"test": dtype_str.upper()},
        )

        try:
            st = Safetensors(path)
            actual = st.tensor_data["test"]
            assert actual.dtype == np_dtype
            np.testing.assert_array_almost_equal(actual, expected)
            st.close()
        finally:
            os.unlink(path)

    def test_load_skips_metadata(self):
        """Verify __metadata__ key is skipped."""
        header = {
            "__metadata__": {"format": "pt"},
            "tensor": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8],
            },
        }

        data = np.array([1.0, 2.0], dtype=np.float32).tobytes()
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
            f.write(data)
            path = f.name

        try:
            st = Safetensors(path)
            assert "__metadata__" not in st.tensor_data
            assert "tensor" in st.tensor_data
            st.close()
        finally:
            os.unlink(path)

    def test_memory_mapping_zero_copy(self):
        """Verify tensors are views into mmap (zero-copy)."""
        expected = np.random.randn(100, 100).astype(np.float32)
        path = self.create_test_file({"test": expected})

        try:
            st = Safetensors(path)
            tensor = st.tensor_data["test"]

            # Check if it's a view (base should be the mmap buffer)
            assert tensor.base is not None, "Tensor should be a view"

            # Modifying original file through mmap should reflect in tensor
            # (This is advanced; basic check is that base exists)
            st.close()
        finally:
            os.unlink(path)

    def test_close_releases_resources(self):
        """Test that close() properly releases mmap."""
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = self.create_test_file({"test": expected})

        try:
            st = Safetensors(path)
            assert st._mmap is not None
            st.close()
            assert st._mmap is None
        finally:
            os.unlink(path)

    def test_1d_tensor(self):
        """Load 1D tensor (e.g., bias, layernorm weight)."""
        expected = np.random.randn(576).astype(np.float32)
        path = self.create_test_file({"norm_weight": expected})

        try:
            st = Safetensors(path)
            actual = st.tensor_data["norm_weight"]
            assert actual.shape == (576,)
            np.testing.assert_array_almost_equal(actual, expected)
            st.close()
        finally:
            os.unlink(path)

    def test_large_tensor(self):
        """Load realistically-sized tensor (embedding table)."""
        vocab_size, hidden_size = 49152, 576
        expected = np.random.randn(vocab_size, hidden_size).astype(np.float16)
        path = self.create_test_file({"embed": expected})

        try:
            st = Safetensors(path)
            actual = st.tensor_data["embed"]
            assert actual.shape == (vocab_size, hidden_size)
            assert actual.dtype == np.float16
            st.close()
        finally:
            os.unlink(path)


class TestSafetensorsEdgeCases(TestSafetensorsLoading):
    """Test edge cases and error conditions."""

    def test_empty_tensor(self):
        """Load tensor with zero elements."""
        expected = np.array([], dtype=np.float32).reshape(0, 64)
        path = self.create_test_file({"empty": expected})

        try:
            st = Safetensors(path)
            actual = st.tensor_data["empty"]
            assert actual.shape == (0, 64)
            st.close()
        finally:
            os.unlink(path)

    def test_scalar_tensor(self):
        """Load 0D (scalar) tensor."""
        expected = np.array(42.0, dtype=np.float32)
        path = self.create_test_file({"scalar": expected})

        try:
            st = Safetensors(path)
            actual = st.tensor_data["scalar"]
            assert actual.shape == ()
            assert actual[()] == 42.0
            st.close()
        finally:
            os.unlink(path)

    def test_3d_tensor(self):
        """Load 3D tensor."""
        expected = np.random.randn(2, 3, 4).astype(np.float32)
        path = self.create_test_file({"3d": expected})

        try:
            st = Safetensors(path)
            actual = st.tensor_data["3d"]
            assert actual.shape == (2, 3, 4)
            st.close()
        finally:
            os.unlink(path)
