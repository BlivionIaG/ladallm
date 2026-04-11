"""Unit tests for F5: SwiGLU MLP."""

import numpy as np
import pytest

from ladallm.mlp import sigmoid, swiglu_mlp


class TestSigmoid:
    """Test sigmoid activation function."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should be 0.5."""
        x = np.array([0.0], dtype=np.float32)
        out = sigmoid(x)

        np.testing.assert_array_almost_equal(out, [0.5])

    def test_sigmoid_positive(self):
        """sigmoid of positive values should be > 0.5."""
        x = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        out = sigmoid(x)

        assert np.all(out > 0.5)
        assert np.all(out < 1.0)

    def test_sigmoid_negative(self):
        """sigmoid of negative values should be < 0.5."""
        x = np.array([-1.0, -2.0, -5.0], dtype=np.float32)
        out = sigmoid(x)

        assert np.all(out > 0.0)
        assert np.all(out < 0.5)

    def test_sigmoid_range(self):
        """sigmoid output should always be in (0, 1)."""
        x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        out = sigmoid(x)

        assert np.all(out > 0)
        assert np.all(out < 1)

    def test_sigmoid_large_positive(self):
        """sigmoid of large positive values should approach 1."""
        x = np.array([10.0, 20.0], dtype=np.float32)
        out = sigmoid(x)

        assert np.all(out > 0.9999)

    def test_sigmoid_large_negative(self):
        """sigmoid of large negative values should approach 0."""
        x = np.array([-10.0, -20.0], dtype=np.float32)
        out = sigmoid(x)

        assert np.all(out < 0.0001)


class TestSwiGLUMLP:
    """Test SwiGLU MLP forward pass."""

    def test_output_shape(self):
        """Output should have same shape as input [seq_len, d_model]."""
        seq_len, d_model, d_ff = 10, 576, 1536

        x = np.random.randn(seq_len, d_model).astype(np.float32)
        W_gate = np.random.randn(d_model, d_ff).astype(np.float32)
        W_up = np.random.randn(d_model, d_ff).astype(np.float32)
        W_down = np.random.randn(d_ff, d_model).astype(np.float32)

        out = swiglu_mlp(x, W_gate, W_up, W_down)

        assert out.shape == (seq_len, d_model)

    def test_single_token(self):
        """Should handle single token [1, d_model]."""
        d_model, d_ff = 576, 1536

        x = np.random.randn(1, d_model).astype(np.float32)
        W_gate = np.random.randn(d_model, d_ff).astype(np.float32)
        W_up = np.random.randn(d_model, d_ff).astype(np.float32)
        W_down = np.random.randn(d_ff, d_model).astype(np.float32)

        out = swiglu_mlp(x, W_gate, W_up, W_down)

        assert out.shape == (1, d_model)

    def test_batch_various_lengths(self):
        """Should handle various sequence lengths."""
        d_model, d_ff = 64, 128

        for seq_len in [1, 5, 10, 100]:
            x = np.random.randn(seq_len, d_model).astype(np.float32)
            W_gate = np.random.randn(d_model, d_ff).astype(np.float32)
            W_up = np.random.randn(d_model, d_ff).astype(np.float32)
            W_down = np.random.randn(d_ff, d_model).astype(np.float32)

            out = swiglu_mlp(x, W_gate, W_up, W_down)

            assert out.shape == (seq_len, d_model)

    def test_deterministic(self):
        """Same input should produce same output."""
        np.random.seed(42)
        seq_len, d_model, d_ff = 5, 64, 128

        x = np.random.randn(seq_len, d_model).astype(np.float32)
        W_gate = np.random.randn(d_model, d_ff).astype(np.float32)
        W_up = np.random.randn(d_model, d_ff).astype(np.float32)
        W_down = np.random.randn(d_ff, d_model).astype(np.float32)

        out1 = swiglu_mlp(x, W_gate, W_up, W_down)
        out2 = swiglu_mlp(x, W_gate, W_up, W_down)

        np.testing.assert_array_equal(out1, out2)

    def test_no_nan(self):
        """Output should not contain NaN or Inf."""
        seq_len, d_model, d_ff = 10, 576, 1536

        x = np.random.randn(seq_len, d_model).astype(np.float32)
        W_gate = np.random.randn(d_model, d_ff).astype(np.float32)
        W_up = np.random.randn(d_model, d_ff).astype(np.float32)
        W_down = np.random.randn(d_ff, d_model).astype(np.float32)

        out = swiglu_mlp(x, W_gate, W_up, W_down)

        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_zero_weights(self):
        """Zero weights should produce zero output (with zero input)."""
        seq_len, d_model, d_ff = 5, 64, 128

        x = np.zeros((seq_len, d_model), dtype=np.float32)
        W_gate = np.zeros((d_model, d_ff), dtype=np.float32)
        W_up = np.zeros((d_model, d_ff), dtype=np.float32)
        W_down = np.zeros((d_ff, d_model), dtype=np.float32)

        out = swiglu_mlp(x, W_gate, W_up, W_down)

        np.testing.assert_array_almost_equal(out, np.zeros((seq_len, d_model)))


class TestSmolLM2Config:
    """Test with SmolLM2-135M specific configuration."""

    def test_smollm2_dimensions(self):
        """Test with SmolLM2 d_model=576, d_ff=1536."""
        d_model = 576
        d_ff = 1536

        seq_len = 128
        x = np.random.randn(seq_len, d_model).astype(np.float32)
        W_gate = np.random.randn(d_model, d_ff).astype(np.float32)
        W_up = np.random.randn(d_model, d_ff).astype(np.float32)
        W_down = np.random.randn(d_ff, d_model).astype(np.float32)

        out = swiglu_mlp(x, W_gate, W_up, W_down)

        assert out.shape == (seq_len, d_model)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
