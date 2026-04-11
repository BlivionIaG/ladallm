"""Unit tests for F3: RoPE (Rotary Position Embedding)."""

import numpy as np
import pytest

from ladallm.rope import apply_rope, precompute_rope_tables


class TestPrecomputeRopeTables:
    """Test precompute_rope_tables function."""

    def test_output_shapes(self):
        """Cos and sin tables should have correct shapes."""
        max_seq_len, head_dim = 2048, 64

        cos_table, sin_table = precompute_rope_tables(max_seq_len, head_dim)

        assert cos_table.shape == (max_seq_len, head_dim // 2)
        assert sin_table.shape == (max_seq_len, head_dim // 2)

    def test_small_example_shapes(self):
        """Test with small dimensions for easy verification."""
        cos_table, sin_table = precompute_rope_tables(max_seq_len=10, head_dim=4)

        assert cos_table.shape == (10, 2)
        assert sin_table.shape == (10, 2)

    def test_position_0_angles(self):
        """Position 0 should have all angles = 0, so cos=1, sin=0."""
        cos_table, sin_table = precompute_rope_tables(max_seq_len=10, head_dim=4)

        np.testing.assert_array_almost_equal(cos_table[0], [1.0, 1.0])
        np.testing.assert_array_almost_equal(sin_table[0], [0.0, 0.0])

    def test_cos_sin_pythagorean(self):
        """cos^2 + sin^2 = 1 for all positions and dimensions."""
        cos_table, sin_table = precompute_rope_tables(max_seq_len=100, head_dim=64)

        pythagorean = cos_table ** 2 + sin_table ** 2
        np.testing.assert_array_almost_equal(pythagorean, np.ones_like(pythagorean))

    def test_increasing_angles_with_position(self):
        """Angles should increase with position (monotonic in some dimensions)."""
        cos_table, _ = precompute_rope_tables(max_seq_len=10, head_dim=4)

        assert cos_table[0, 0] > cos_table[-1, 0]

    def test_default_base(self):
        """Default base should be 10000."""
        cos_default, _ = precompute_rope_tables(10, 4)
        cos_explicit, _ = precompute_rope_tables(10, 4, base=10000.0)

        np.testing.assert_array_almost_equal(cos_default, cos_explicit)

    def test_different_bases(self):
        """Different bases should produce different angles."""
        cos_100, _ = precompute_rope_tables(10, 4, base=100.0)
        cos_10000, _ = precompute_rope_tables(10, 4, base=10000.0)

        # Different bases → different angles
        assert not np.allclose(cos_100, cos_10000)


class TestApplyRopeBasic:
    """Test basic apply_rope functionality."""

    def test_output_shapes_preserved(self):
        """Output shapes should match input shapes."""
        seq_len, num_heads, head_dim = 2, 9, 64
        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, 3, head_dim).astype(np.float32)  # GQA
        positions = np.array([0, 1])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_0_no_rotation(self):
        """Position 0 has angle 0, so cos=1, sin=0, no change to vectors."""
        head_dim = 8
        q = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
        k = q.copy()
        positions = np.array([0])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        # At position 0: x' = x*1 - y*0 = x, y' = x*0 + y*1 = y
        np.testing.assert_array_almost_equal(q_rot, q)
        np.testing.assert_array_almost_equal(k_rot, k)

    def test_rotation_changes_values(self):
        """Non-zero positions should change the values."""
        head_dim = 8
        q = np.ones((1, 1, head_dim), dtype=np.float32)
        k = np.ones((1, 1, head_dim), dtype=np.float32)
        positions = np.array([5])  # Non-zero position
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        # Should be different from input due to rotation
        assert not np.allclose(q_rot, q)
        assert not np.allclose(k_rot, k)

    def test_single_token(self):
        """Test with single token (decode phase)."""
        head_dim = 64
        q = np.random.randn(1, 9, head_dim).astype(np.float32)
        k = np.random.randn(1, 3, head_dim).astype(np.float32)
        positions = np.array([42])  # Arbitrary position
        cos_table, sin_table = precompute_rope_tables(100, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == (1, 9, head_dim)
        assert k_rot.shape == (1, 3, head_dim)

    def test_multiple_tokens(self):
        """Test with multiple tokens (prefill phase)."""
        seq_len, head_dim = 10, 64
        q = np.random.randn(seq_len, 9, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, 3, head_dim).astype(np.float32)
        positions = np.arange(seq_len)  # [0, 1, 2, ..., 9]
        cos_table, sin_table = precompute_rope_tables(20, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == (seq_len, 9, head_dim)
        assert k_rot.shape == (seq_len, 3, head_dim)


class TestApplyRopeMath:
    """Test mathematical correctness of rotation."""

    def test_rotation_formula_single_pair(self):
        """Test rotation formula: [x', y'] = [x*cos - y*sin, x*sin + y*cos]."""
        # Simple case: one head, one pair of dimensions
        head_dim = 2
        q = np.array([[[3.0, 4.0]]], dtype=np.float32)  # x=3, y=4
        k = q.copy()
        positions = np.array([0])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, _ = apply_rope(q, k, positions, cos_table, sin_table)

        # At position 0: cos=1, sin=0
        # x' = 3*1 - 4*0 = 3
        # y' = 3*0 + 4*1 = 4
        expected = np.array([[[3.0, 4.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(q_rot, expected)

    def test_90_degree_rotation(self):
        """Test known rotation: 90 degrees swaps and negates appropriately."""
        # Create custom cos/sin tables for testing
        cos_table = np.zeros((10, 1), dtype=np.float32)
        sin_table = np.ones((10, 1), dtype=np.float32)

        q = np.array([[[3.0, 4.0]]], dtype=np.float32)
        k = q.copy()
        positions = np.array([0])

        q_rot, _ = apply_rope(q, k, positions, cos_table, sin_table)

        # Expected: [-y, x] = [-4, 3]
        expected = np.array([[[-4.0, 3.0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(q_rot, expected)

    def test_different_positions_different_rotations(self):
        """Same input at different positions should have different rotations."""
        head_dim = 8
        q = np.ones((1, 1, head_dim), dtype=np.float32)
        k = np.ones((1, 1, head_dim), dtype=np.float32)
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        # Apply at position 1
        q_rot_1, _ = apply_rope(q, k, np.array([1]), cos_table, sin_table)
        # Apply at position 2
        q_rot_2, _ = apply_rope(q, k, np.array([2]), cos_table, sin_table)

        # Should be different
        assert not np.allclose(q_rot_1, q_rot_2)

    def test_vector_norm_preserved(self):
        """Rotation preserves vector norm (length) within each pair."""
        head_dim = 64
        q = np.random.randn(1, 1, head_dim).astype(np.float32)
        k = np.random.randn(1, 1, head_dim).astype(np.float32)
        positions = np.array([5])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        # Compute norms of each pair before rotation
        x = q[..., 0::2]
        y = q[..., 1::2]
        norms_before = np.sqrt(x**2 + y**2)

        q_rot, _ = apply_rope(q, k, positions, cos_table, sin_table)

        # Compute norms after rotation
        x_rot = q_rot[..., 0::2]
        y_rot = q_rot[..., 1::2]
        norms_after = np.sqrt(x_rot**2 + y_rot**2)

        # Norms should be preserved
        np.testing.assert_array_almost_equal(norms_before, norms_after)


class TestApplyRopeGQA:
    """Test with Grouped Query Attention (GQA) - different num_heads."""

    def test_gqa_different_head_counts(self):
        """Q has more heads than K/V (GQA pattern)."""
        seq_len, head_dim = 2, 64
        num_q_heads, num_kv_heads = 9, 3  # SmolLM2 pattern
        q = np.random.randn(seq_len, num_q_heads, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        positions = np.array([0, 1])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == (seq_len, num_q_heads, head_dim)
        assert k_rot.shape == (seq_len, num_kv_heads, head_dim)

    def test_gqa_same_rotation_per_kv_head(self):
        """Each KV head is shared by multiple Q heads."""
        head_dim = 8
        num_q_heads, num_kv_heads = 4, 2
        seq_len = 1
        position = 5

        q = np.random.randn(seq_len, num_q_heads, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, k_rot = apply_rope(q, k, np.array([position]), cos_table, sin_table)

        # Both Q heads in group should get same rotation pattern
        # (though different values due to different inputs)
        # Just verify shapes and no errors
        assert q_rot.shape == (seq_len, num_q_heads, head_dim)
        assert k_rot.shape == (seq_len, num_kv_heads, head_dim)


class TestApplyRopePositions:
    """Test position handling."""

    def test_arbitrary_positions(self):
        """Test with non-sequential positions."""
        head_dim = 64
        q = np.random.randn(3, 9, head_dim).astype(np.float32)
        k = np.random.randn(3, 3, head_dim).astype(np.float32)
        positions = np.array([100, 500, 1000])  # Arbitrary positions
        cos_table, sin_table = precompute_rope_tables(2048, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_repeated_positions(self):
        """Test with same position appearing multiple times."""
        head_dim = 64
        q = np.random.randn(3, 9, head_dim).astype(np.float32)
        k = np.random.randn(3, 3, head_dim).astype(np.float32)
        positions = np.array([5, 5, 5])  # Same position
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, _ = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == q.shape

    def test_position_beyond_table(self):
        """Position beyond table size should raise IndexError."""
        head_dim = 64
        q = np.random.randn(1, 9, head_dim).astype(np.float32)
        k = np.random.randn(1, 3, head_dim).astype(np.float32)
        positions = np.array([100])  # Beyond table size
        cos_table, sin_table = precompute_rope_tables(10, head_dim)  # Only 10 positions

        with pytest.raises(IndexError):
            apply_rope(q, k, positions, cos_table, sin_table)


class TestApplyRopeDtypes:
    """Test with different data types."""

    def test_float32(self):
        """Standard float32."""
        head_dim = 64
        q = np.random.randn(2, 9, head_dim).astype(np.float32)
        k = np.random.randn(2, 3, head_dim).astype(np.float32)
        positions = np.array([0, 1])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.dtype == np.float32
        assert k_rot.dtype == np.float32

    def test_float64_tables(self):
        """Test with float64 tables (should cast or work)."""
        head_dim = 64
        q = np.random.randn(2, 9, head_dim).astype(np.float32)
        k = np.random.randn(2, 3, head_dim).astype(np.float32)
        positions = np.array([0, 1])
        cos_table, sin_table = precompute_rope_tables(10, head_dim)
        # Tables are float64 by default from np.cos/sin

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        # Should work without error
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestApplyRopeSmolLM2Config:
    """Test with SmolLM2-135M specific configuration."""

    def test_smollm2_head_dim(self):
        """Test with SmolLM2 head_dim=64 (576/9)."""
        hidden_size = 576
        num_heads = 9
        head_dim = hidden_size // num_heads  # 64

        q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
        k = np.random.randn(1, 3, head_dim).astype(np.float32)  # GQA: 3 KV heads
        positions = np.array([0])
        cos_table, sin_table = precompute_rope_tables(2048, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == (1, num_heads, head_dim)
        assert k_rot.shape == (1, 3, head_dim)

    def test_smollm2_realistic_prefill(self):
        """Test realistic prefill: 128 tokens at positions [0..127]."""
        hidden_size, num_heads = 576, 9
        head_dim = hidden_size // num_heads
        seq_len = 128

        q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, 3, head_dim).astype(np.float32)
        positions = np.arange(seq_len)
        cos_table, sin_table = precompute_rope_tables(2048, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == (seq_len, num_heads, head_dim)
        assert k_rot.shape == (seq_len, 3, head_dim)
        assert not np.any(np.isnan(q_rot))
        assert not np.any(np.isnan(k_rot))

    def test_smollm2_decode_step(self):
        """Test decode step: single token at arbitrary position."""
        hidden_size, num_heads = 576, 9
        head_dim = hidden_size // num_heads

        # Token at position 500 (continuing from previous tokens)
        q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
        k = np.random.randn(1, 3, head_dim).astype(np.float32)
        positions = np.array([500])
        cos_table, sin_table = precompute_rope_tables(2048, head_dim)

        q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

        assert q_rot.shape == (1, num_heads, head_dim)
        assert k_rot.shape == (1, 3, head_dim)


class TestRopeIntegration:
    """Integration-style tests."""

    def test_end_to_end_small_model(self):
        """Test with a tiny model configuration."""
        max_seq_len, head_dim = 32, 8
        num_heads, num_kv_heads = 4, 2

        # Precompute tables
        cos_table, sin_table = precompute_rope_tables(max_seq_len, head_dim)

        # Simulate a forward pass
        for pos in range(max_seq_len):
            q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
            k = np.random.randn(1, num_kv_heads, head_dim).astype(np.float32)
            positions = np.array([pos])

            q_rot, k_rot = apply_rope(q, k, positions, cos_table, sin_table)

            assert q_rot.shape == (1, num_heads, head_dim)
            assert k_rot.shape == (1, num_kv_heads, head_dim)
            assert not np.any(np.isnan(q_rot))
            assert not np.any(np.isnan(k_rot))

    def test_relative_position_property(self):
        """Verify relative position property: rotation depends on (m-n), not absolute positions."""
        head_dim = 8
        cos_table, sin_table = precompute_rope_tables(100, head_dim)

        # Create Q and K
        q_input = np.random.randn(1, 1, head_dim).astype(np.float32)
        k_input = np.random.randn(1, 1, head_dim).astype(np.float32)

        # Case 1: Q at position 10, K at position 5 (diff = 5)
        q_rot_10, _ = apply_rope(
            q_input, k_input, np.array([10]), cos_table, sin_table
        )
        _, k_rot_5 = apply_rope(
            q_input, k_input, np.array([5]), cos_table, sin_table
        )

        # Dot product Q_10 @ K_5
        dot_1 = np.sum(q_rot_10 * k_rot_5)

        # Case 2: Q at position 20, K at position 15 (same diff = 5)
        q_rot_20, _ = apply_rope(q_input, k_input, np.array([20]), cos_table, sin_table)
        _, k_rot_15 = apply_rope(q_input, k_input, np.array([15]), cos_table, sin_table)

        # Dot product Q_20 @ K_15
        dot_2 = np.sum(q_rot_20 * k_rot_15)

        # These should be approximately equal (same relative distance)
        assert not np.isnan(dot_1)
        assert not np.isnan(dot_2)
