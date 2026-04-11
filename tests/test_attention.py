"""Unit tests for F4: Attention + KV-cache reads."""

import numpy as np

from ladallm.attention import (
    attention_forward,
    causal_mask,
    compute_qkv,
    softmax,
)


class TestSoftmax:
    """Test softmax function."""

    def test_sums_to_one(self):
        """Softmax output should sum to 1 along specified axis."""
        x = np.random.randn(5, 10).astype(np.float32)
        out = softmax(x, axis=-1)

        np.testing.assert_array_almost_equal(np.sum(out, axis=-1), np.ones(5))

    def test_numerical_stability(self):
        """Should handle large values without overflow."""
        x = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32)
        out = softmax(x, axis=-1)

        expected = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(out, expected, decimal=5)

    def test_negative_values(self):
        """Should handle negative values correctly."""
        x = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        out = softmax(x, axis=-1)

        assert np.all(out > 0)
        assert np.all(out < 1)
        np.testing.assert_array_almost_equal(np.sum(out), 1.0)

    def test_single_element(self):
        """Softmax of single element should be 1."""
        x = np.array([[5.0]], dtype=np.float32)
        out = softmax(x, axis=-1)

        np.testing.assert_array_almost_equal(out, [[1.0]])


class TestComputeQKV:
    """Test QKV projection."""

    def test_output_shapes(self):
        """Q, K, V should have correct shapes after projection."""
        seq_len, hidden_size = 10, 576
        num_heads, num_kv_heads, head_dim = 9, 3, 64

        x = np.random.randn(seq_len, hidden_size).astype(np.float32)
        w_q = np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32)
        w_k = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)
        w_v = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)

        q, k, v = compute_qkv(x, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim)

        assert q.shape == (seq_len, num_heads, head_dim)
        assert k.shape == (seq_len, num_kv_heads, head_dim)
        assert v.shape == (seq_len, num_kv_heads, head_dim)

    def test_single_token(self):
        """Should handle single token (decode phase)."""
        hidden_size = 576
        num_heads, num_kv_heads, head_dim = 9, 3, 64

        x = np.random.randn(1, hidden_size).astype(np.float32)
        w_q = np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32)
        w_k = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)
        w_v = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)

        q, k, v = compute_qkv(x, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim)

        assert q.shape == (1, num_heads, head_dim)
        assert k.shape == (1, num_kv_heads, head_dim)
        assert v.shape == (1, num_kv_heads, head_dim)

    def test_different_head_counts(self):
        """Should work with GQA (different num_heads vs num_kv_heads)."""
        seq_len, hidden_size = 5, 256
        num_heads, num_kv_heads, head_dim = 8, 2, 32

        x = np.random.randn(seq_len, hidden_size).astype(np.float32)
        w_q = np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32)
        w_k = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)
        w_v = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)

        q, k, v = compute_qkv(x, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim)

        assert q.shape == (seq_len, num_heads, head_dim)
        assert k.shape == (seq_len, num_kv_heads, head_dim)
        assert v.shape == (seq_len, num_kv_heads, head_dim)


class TestCausalMask:
    """Test causal mask creation."""

    def test_lower_triangle_zero(self):
        """Lower triangle (including diagonal) should be 0."""
        mask = causal_mask(5, 5)

        for i in range(5):
            for j in range(i + 1):
                assert mask[i, j] == 0.0

    def test_upper_triangle_neg_inf(self):
        """Upper triangle should be -inf."""
        mask = causal_mask(5, 5)

        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j] == float("-inf")

    def test_rectangular_mask(self):
        """Should handle query length != key length."""
        mask = causal_mask(3, 5)

        assert mask.shape == (3, 5)
        # Row 0: [0, -inf, -inf, -inf, -inf]
        assert mask[0, 0] == 0.0
        assert mask[0, 1] == float("-inf")

    def test_mask_applied_to_scores(self):
        """Mask should work when added to attention scores."""
        mask = causal_mask(3, 3)
        scores = np.random.randn(3, 3).astype(np.float32)

        masked_scores = scores + mask

        # Upper triangle should be -inf
        for i in range(3):
            for j in range(i + 1, 3):
                assert masked_scores[i, j] == float("-inf")


class TestAttentionForward:
    """Test attention computation."""

    def test_output_shape_prefill(self):
        """Attention output should have correct shape for prefill."""
        seq_len, num_heads, head_dim = 10, 9, 64
        num_kv_heads = 3

        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        attn_scale = 1.0 / np.sqrt(head_dim)

        out = attention_forward(q, k_cache, v_cache, head_dim, num_kv_heads, num_heads, attn_scale)

        assert out.shape == (seq_len, num_heads, head_dim)

    def test_output_shape_decode(self):
        """Attention output should have correct shape for decode."""
        num_heads, head_dim = 9, 64
        num_kv_heads = 3
        cache_len = 50

        q = np.random.randn(num_heads, head_dim).astype(np.float32)
        k_cache = np.random.randn(cache_len, num_kv_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(cache_len, num_kv_heads, head_dim).astype(np.float32)
        attn_scale = 1.0 / np.sqrt(head_dim)

        out = attention_forward(q, k_cache, v_cache, head_dim, num_kv_heads, num_heads, attn_scale)

        assert out.shape == (1, num_heads, head_dim)

    def test_gqa_tiling(self):
        """GQA: K and V should be repeated to match Q head count."""
        seq_len, num_heads, head_dim = 5, 6, 32
        num_kv_heads = 2

        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k_cache = np.ones((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        v_cache = np.ones((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        attn_scale = 1.0 / np.sqrt(head_dim)

        out = attention_forward(q, k_cache, v_cache, num_kv_heads, num_heads, attn_scale)

        # GQA tiling happens internally, output should be correct shape
        assert out.shape == (seq_len, num_heads, head_dim)

    def test_with_causal_mask(self):
        """Causal mask should prevent attending to future positions."""
        seq_len, num_heads, head_dim = 4, 2, 16
        num_kv_heads = 2

        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        mask = causal_mask(seq_len, seq_len)
        attn_scale = 1.0 / np.sqrt(head_dim)

        out = attention_forward(
            q, k_cache, v_cache, num_kv_heads, num_heads, attn_scale, mask=mask
        )

        assert out.shape == (seq_len, num_heads, head_dim)
        assert not np.any(np.isnan(out))

    def test_no_nan_with_mask(self):
        """Masked attention should not produce NaN."""
        seq_len, num_heads, head_dim = 8, 4, 32
        num_kv_heads = 2

        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        mask = causal_mask(seq_len, seq_len)
        attn_scale = 1.0 / np.sqrt(head_dim)

        out = attention_forward(
            q, k_cache, v_cache, num_kv_heads, num_heads, attn_scale, mask=mask
        )

        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestAttentionIntegration:
    """Integration tests for full attention pipeline."""

    def test_prefill_pipeline(self):
        """Full prefill: QKV -> RoPE -> Cache -> Attention."""
        from ladallm.rope import apply_rope, precompute_rope_tables

        seq_len, hidden_size = 10, 576
        num_heads, num_kv_heads, head_dim = 9, 3, 64

        # Setup
        x = np.random.randn(seq_len, hidden_size).astype(np.float32)
        w_q = np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32)
        w_k = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)
        w_v = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)

        positions = np.arange(seq_len)
        cos_table, sin_table = precompute_rope_tables(seq_len, head_dim)
        mask = causal_mask(seq_len, seq_len)
        attn_scale = 1.0 / np.sqrt(head_dim)

        # Pipeline
        q, k, v = compute_qkv(x, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim)
        q, k = apply_rope(q, k, positions, cos_table, sin_table)

        # Use K, V as "cache" for simplicity
        out = attention_forward(q, k, v, num_kv_heads, num_heads, attn_scale, mask=mask)

        assert out.shape == (seq_len, num_heads, head_dim)
        assert not np.any(np.isnan(out))

    def test_decode_pipeline(self):
        """Full decode: single token with cached history."""
        from ladallm.rope import apply_rope, precompute_rope_tables

        cache_len = 50
        hidden_size = 576
        num_heads, num_kv_heads, head_dim = 9, 3, 64

        # Setup
        x_new = np.random.randn(1, hidden_size).astype(np.float32)
        w_q = np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32)
        w_k = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)
        w_v = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)

        # Simulate cached K, V
        k_cache = np.random.randn(cache_len, num_kv_heads, head_dim).astype(np.float32)
        v_cache = np.random.randn(cache_len, num_kv_heads, head_dim).astype(np.float32)

        positions = np.array([cache_len])  # New position
        cos_table, sin_table = precompute_rope_tables(cache_len + 1, head_dim)
        attn_scale = 1.0 / np.sqrt(head_dim)

        # Pipeline
        q, k, _ = compute_qkv(x_new, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim)
        q, k = apply_rope(q, k, positions, cos_table, sin_table)

        # Decode: no mask needed
        out = attention_forward(
            q, k_cache, v_cache, num_kv_heads, num_heads, attn_scale, mask=None
        )

        assert out.shape == (1, num_heads, head_dim)


class TestSmolLM2Config:
    """Test with SmolLM2-135M specific configuration."""

    def test_smollm2_dimensions(self):
        """Test with SmolLM2 hidden_size=576, num_heads=9, num_kv_heads=3."""
        hidden_size = 576
        num_heads = 9
        num_kv_heads = 3
        head_dim = hidden_size // num_heads  # 64

        seq_len = 128
        x = np.random.randn(seq_len, hidden_size).astype(np.float32)
        w_q = np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32)
        w_k = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)
        w_v = np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32)

        q, k, v = compute_qkv(x, w_q, w_k, w_v, num_heads, num_kv_heads, head_dim)

        assert q.shape == (seq_len, num_heads, head_dim)
        assert k.shape == (seq_len, num_kv_heads, head_dim)
        assert v.shape == (seq_len, num_kv_heads, head_dim)

        # Attention
        mask = causal_mask(seq_len, seq_len)
        attn_scale = 1.0 / np.sqrt(head_dim)
        out = attention_forward(q, k, v, num_kv_heads, num_heads, attn_scale, mask=mask)

        assert out.shape == (seq_len, num_heads, head_dim)
