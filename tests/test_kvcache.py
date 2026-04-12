"""Unit tests for F7: Naive KV Cache.

Tests the NaiveKVCache class which provides per-layer storage for Key and Value
tensors during autoregressive generation.
"""

import numpy as np
import pytest

from ladallm.kvcache import NaiveKVCache, create_layer_caches


class TestNaiveKVCacheInitialization:
    """Test cache initialization and properties."""

    def test_basic_initialization(self):
        """Cache should initialize with correct dimensions."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        assert cache.max_seq_len == 100
        assert cache.num_kv_heads == 3
        assert cache.head_dim == 64
        assert len(cache) == 0

    def test_initial_buffers_zero_length(self):
        """Cache buffers should exist but have zero length initially."""
        cache = NaiveKVCache(max_seq_len=2048, num_kv_heads=4, head_dim=128)

        k, v = cache.get()
        assert k.shape == (0, 4, 128)
        assert v.shape == (0, 4, 128)

    def test_memory_usage_calculation(self):
        """Memory usage should account for both K and V buffers."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        expected_bytes = 2 * 100 * 3 * 64 * 4  # 2 tensors * floats
        assert cache.memory_usage_bytes == expected_bytes


class TestNaiveKVCacheAppend:
    """Test appending K/V tensors to cache."""

    def test_append_single_token(self):
        """Should append single token and update length."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)

        cache.append(k, v)

        assert len(cache) == 1
        k_cached, v_cached = cache.get()
        assert k_cached.shape == (1, 3, 64)
        np.testing.assert_array_equal(k_cached, k)
        np.testing.assert_array_equal(v_cached, v)

    def test_append_multiple_tokens_prefill(self):
        """Should append multiple tokens (prefill scenario)."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        k = np.random.randn(10, 3, 64).astype(np.float32)
        v = np.random.randn(10, 3, 64).astype(np.float32)

        cache.append(k, v)

        assert len(cache) == 10
        k_cached, v_cached = cache.get()
        assert k_cached.shape == (10, 3, 64)

    def test_append_sequential_decode(self):
        """Should append tokens one at a time (decode scenario)."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        # Prefill with 5 tokens
        k_prefill = np.random.randn(5, 3, 64).astype(np.float32)
        v_prefill = np.random.randn(5, 3, 64).astype(np.float32)
        cache.append(k_prefill, v_prefill)

        # Decode: add 3 more tokens one at a time
        for i in range(3):
            k_new = np.random.randn(1, 3, 64).astype(np.float32)
            v_new = np.random.randn(1, 3, 64).astype(np.float32)
            cache.append(k_new, v_new)

        assert len(cache) == 8
        k_cached, v_cached = cache.get()
        assert k_cached.shape == (8, 3, 64)

        # Verify prefill data is intact
        np.testing.assert_array_equal(k_cached[:5], k_prefill)
        np.testing.assert_array_equal(v_cached[:5], v_prefill)


class TestNaiveKVCacheGet:
    """Test retrieving cached K/V tensors."""

    def test_get_returns_views(self):
        """get() should return views, not copies."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        k = np.random.randn(5, 3, 64).astype(np.float32)
        v = np.random.randn(5, 3, 64).astype(np.float32)
        cache.append(k, v)

        k_cached, v_cached = cache.get()

        # Modifying returned arrays should affect cache (they're views)
        k_cached[0, 0, 0] = 999.0
        assert cache.k[0, 0, 0] == 999.0

    def test_get_returns_correct_subset(self):
        """get() should only return up to current length."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        # Append 5 tokens
        k = np.random.randn(5, 3, 64).astype(np.float32)
        v = np.random.randn(5, 3, 64).astype(np.float32)
        cache.append(k, v)

        k_cached, v_cached = cache.get()

        # Should only be 5 positions
        assert k_cached.shape == (5, 3, 64)
        # Pre-allocated buffer is 100, but we only get 5
        assert cache.k.shape == (100, 3, 64)


class TestNaiveKVCacheValidation:
    """Test input validation and error handling."""

    def test_rejects_mismatched_k_v_shapes(self):
        """Should raise error if k and v have different shapes."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        k = np.random.randn(5, 3, 64).astype(np.float32)
        v = np.random.randn(5, 3, 32).astype(np.float32)  # Wrong head_dim

        with pytest.raises(ValueError, match="shapes must match"):
            cache.append(k, v)

    def test_rejects_wrong_num_kv_heads(self):
        """Should raise error if num_kv_heads doesn't match cache."""
        cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)

        k = np.random.randn(5, 4, 64).astype(np.float32)  # Wrong num_kv_heads
        v = np.random.randn(5, 4, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Expected K,V shape"):
            cache.append(k, v)

    def test_rejects_overflow(self):
        """Should raise error when exceeding max_seq_len."""
        cache = NaiveKVCache(max_seq_len=10, num_kv_heads=3, head_dim=64)

        # Fill cache to capacity
        k = np.random.randn(10, 3, 64).astype(np.float32)
        v = np.random.randn(10, 3, 64).astype(np.float32)
        cache.append(k, v)

        # Try to append one more
        k_new = np.random.randn(1, 3, 64).astype(np.float32)
        v_new = np.random.randn(1, 3, 64).astype(np.float32)

        with pytest.raises(RuntimeError, match="Cache overflow"):
            cache.append(k_new, v_new)

    def test_rejects_negative_dimensions(self):
        """Should raise error for non-positive dimensions."""
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            NaiveKVCache(max_seq_len=0, num_kv_heads=3, head_dim=64)

        with pytest.raises(ValueError, match="num_kv_heads must be positive"):
            NaiveKVCache(max_seq_len=100, num_kv_heads=0, head_dim=64)

        with pytest.raises(ValueError, match="head_dim must be positive"):
            NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=-1)


class TestNaiveKVCacheSmolLM2Config:
    """Test with SmolLM2-135M specific configuration."""

    def test_smollm2_dimensions(self):
        """Test with SmolLM2 parameters: 3 KV heads, 64 head_dim."""
        max_seq_len = 2048
        num_kv_heads = 3
        head_dim = 64

        cache = NaiveKVCache(
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Simulate a typical sequence
        prompt_len = 128
        k_prefill = np.random.randn(prompt_len, num_kv_heads, head_dim).astype(
            np.float32
        )
        v_prefill = np.random.randn(prompt_len, num_kv_heads, head_dim).astype(
            np.float32
        )

        cache.append(k_prefill, v_prefill)
        assert len(cache) == prompt_len

        # Generate 20 tokens
        for _ in range(20):
            k_new = np.random.randn(1, num_kv_heads, head_dim).astype(np.float32)
            v_new = np.random.randn(1, num_kv_heads, head_dim).astype(np.float32)
            cache.append(k_new, v_new)

        assert len(cache) == prompt_len + 20
        k_cached, v_cached = cache.get()
        assert k_cached.shape == (148, 3, 64)


class TestCreateLayerCaches:
    """Test the create_layer_caches helper function."""

    def test_creates_correct_number_of_caches(self):
        """Should create one cache per layer."""
        caches = create_layer_caches(
            max_seq_len=100,
            num_layers=30,
            num_kv_heads=3,
            head_dim=64,
        )

        assert len(caches) == 30
        assert all(isinstance(c, NaiveKVCache) for c in caches)

    def test_each_cache_is_independent(self):
        """Each layer's cache should be independent."""
        caches = create_layer_caches(
            max_seq_len=100,
            num_layers=3,
            num_kv_heads=3,
            head_dim=64,
        )

        # Append to layer 0 only
        k = np.random.randn(5, 3, 64).astype(np.float32)
        v = np.random.randn(5, 3, 64).astype(np.float32)
        caches[0].append(k, v)

        assert len(caches[0]) == 5
        assert len(caches[1]) == 0  # Unchanged
        assert len(caches[2]) == 0  # Unchanged

    def test_smollm2_full_cache_set(self):
        """Test creating full cache set for SmolLM2-135M."""
        caches = create_layer_caches(
            max_seq_len=2048,
            num_layers=30,  # SmolLM2 has 30 layers
            num_kv_heads=3,
            head_dim=64,
        )

        # Verify total memory usage
        total_bytes = sum(c.memory_usage_bytes for c in caches)
        expected_per_layer = 2 * 2048 * 3 * 64 * 4  # bytes
        expected_total = 30 * expected_per_layer

        assert total_bytes == expected_total

        # Should be ~90MB for full cache
        assert total_bytes / (1024 * 1024) == pytest.approx(90.0, rel=0.01)
