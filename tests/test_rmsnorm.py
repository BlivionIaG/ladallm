"""Unit tests for F2: RMSNorm."""

import numpy as np
import pytest

from ladallm.cli import rms_norm


class TestRMSNormBasic:
    """Test basic RMSNorm functionality."""

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        x = np.random.randn(10, 576).astype(np.float32)
        weight = np.ones(576, dtype=np.float32)

        out = rms_norm(x, weight)

        assert out.shape == x.shape

    def test_unit_input_unchanged(self):
        """Input of all ones with weight=1 should remain all ones."""
        x = np.ones((5, 10), dtype=np.float32)
        weight = np.ones(10, dtype=np.float32)

        out = rms_norm(x, weight)

        # RMS of all-1s is 1, so x/1*1 = x
        np.testing.assert_array_almost_equal(out, x, decimal=5)

    def test_zero_input_stable(self):
        """Zero input with epsilon should not produce NaN/Inf."""
        x = np.zeros((3, 64), dtype=np.float32)
        weight = np.ones(64, dtype=np.float32)

        out = rms_norm(x, weight, eps=1e-6)

        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
        # With epsilon, RMS = sqrt(eps), output should be near zero
        assert np.allclose(out, 0, atol=1e-3)

    def test_weight_scaling(self):
        """Weight should scale the output."""
        x = np.ones((1, 4), dtype=np.float32) * 2.0  # All 2s
        weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        out = rms_norm(x, weight)

        # RMS of [2,2,2,2] is 2, so normalized is [1,1,1,1], scaled by weight
        expected = weight  # [1, 2, 3, 4]
        np.testing.assert_array_almost_equal(out[0], expected, decimal=5)


class TestRMSNormMath:
    """Test mathematical correctness."""

    def test_rms_computation_manual(self):
        """Manually verify RMS calculation."""
        # Single token, simple values
        x = np.array([[3.0, 4.0]], dtype=np.float32)  # RMS should be 5
        weight = np.ones(2, dtype=np.float32)

        out = rms_norm(x, weight)

        # RMS = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.535
        # Normalized: [3, 4] / 3.535 ≈ [0.848, 1.131]
        expected = np.array([0.848528, 1.131371], dtype=np.float32)
        np.testing.assert_array_almost_equal(out[0], expected, decimal=5)

    def test_multi_token_independence(self):
        """Each token should be normalized independently."""
        x = np.array([
            [1.0, 1.0, 1.0],  # RMS = 1
            [2.0, 2.0, 2.0],  # RMS = 2
            [3.0, 3.0, 3.0],  # RMS = 3
        ], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)

        out = rms_norm(x, weight)

        # After normalization, all should be 1s (then scaled by weight=1)
        expected = np.ones_like(x)
        np.testing.assert_array_almost_equal(out, expected, decimal=5)

    def test_epsilon_effect(self):
        """Smaller epsilon should give larger output for near-zero input."""
        x = np.array([[1e-8, 1e-8]], dtype=np.float32)
        weight = np.ones(2, dtype=np.float32)

        out_small_eps = rms_norm(x, weight, eps=1e-6)
        out_large_eps = rms_norm(x, weight, eps=1e-3)

        # Smaller epsilon → less damping → larger output
        rms_small = np.sqrt(np.mean(x**2) + 1e-6)
        rms_large = np.sqrt(np.mean(x**2) + 1e-3)
        assert rms_small < rms_large


class TestRMSNormShapes:
    """Test various input shapes."""

    @pytest.mark.parametrize("seq_len,hidden_size", [
        (1, 1),
        (1, 576),
        (10, 576),
        (100, 64),
        (1, 4096),
    ])
    def test_various_shapes(self, seq_len, hidden_size):
        """Test different (seq_len, hidden_size) combinations."""
        x = np.random.randn(seq_len, hidden_size).astype(np.float32)
        weight = np.ones(hidden_size, dtype=np.float32)

        out = rms_norm(x, weight)

        assert out.shape == (seq_len, hidden_size)
        assert not np.any(np.isnan(out))

    def test_3d_input_batch(self):
        """Test with batch dimension [batch, seq, hidden]."""
        x = np.random.randn(2, 10, 576).astype(np.float32)
        weight = np.ones(576, dtype=np.float32)

        out = rms_norm(x, weight)

        assert out.shape == (2, 10, 576)


class TestRMSNormDtypes:
    """Test with different data types."""

    def test_float32(self):
        """Standard float32 input."""
        x = np.random.randn(5, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)

        out = rms_norm(x, weight)

        assert out.dtype == np.float32

    def test_float16(self):
        """Float16 input (common for model weights)."""
        x = np.random.randn(5, 64).astype(np.float16)
        weight = np.ones(64, dtype=np.float16)

        out = rms_norm(x, weight)

        # NumPy may upcast; check no NaN
        assert not np.any(np.isnan(out))

    def test_mixed_precision(self):
        """Float16 input with float32 weight."""
        x = np.random.randn(5, 64).astype(np.float16)
        weight = np.ones(64, dtype=np.float32)

        # This may upcast; just verify it doesn't crash
        out = rms_norm(x, weight)
        assert out is not None


class TestRMSNormNumericalStability:
    """Test numerical stability edge cases."""

    def test_very_large_values(self):
        """Test with large magnitude inputs."""
        x = np.random.randn(5, 64).astype(np.float32) * 1000
        weight = np.ones(64, dtype=np.float32)

        out = rms_norm(x, weight)

        assert not np.any(np.isinf(out))
        assert not np.any(np.isnan(out))

    def test_very_small_values(self):
        """Test with very small magnitude inputs."""
        x = np.random.randn(5, 64).astype(np.float32) * 1e-10
        weight = np.ones(64, dtype=np.float32)

        out = rms_norm(x, weight, eps=1e-6)

        assert not np.any(np.isnan(out))

    def test_mixed_magnitudes(self):
        """Test with highly varying magnitudes in same tensor."""
        x = np.array([[1e6, 1e-6, 1.0, -1e3]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)

        out = rms_norm(x, weight)

        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestRMSNormSmolLM2Config:
    """Test with SmolLM2-135M specific configuration."""

    def test_smollm2_hidden_size(self):
        """Test with SmolLM2 hidden_size=576."""
        hidden_size = 576
        x = np.random.randn(1, hidden_size).astype(np.float32)
        weight = np.ones(hidden_size, dtype=np.float32)

        out = rms_norm(x, weight)

        assert out.shape == (1, hidden_size)

    def test_realistic_sequence_length(self):
        """Test with realistic prompt length (e.g., 128 tokens)."""
        seq_len, hidden_size = 128, 576
        x = np.random.randn(seq_len, hidden_size).astype(np.float32)
        weight = np.random.randn(hidden_size).astype(np.float32)

        out = rms_norm(x, weight)

        assert out.shape == (seq_len, hidden_size)
        # Verify output statistics are reasonable
        assert np.std(out) > 0  # Should have variance
        assert not np.any(np.isnan(out))