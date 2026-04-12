"""Unit tests for F6: Decoder block + full forward."""

import numpy as np
import pytest

from ladallm.model import LlamaDecoderBlock, LlamaModel
from ladallm.kvcache import KVCache
from ladallm.safetensors import Safetensors


class TestLlamaDecoderBlockShapes:
    """Test output shapes of decoder block."""

    def test_output_shape_prefill(self):
        """Output should preserve input shape [seq_len, hidden_size] during prefill."""
        seq_len, hidden_size = 10, 576
        num_heads, num_kv_heads = 9, 3
        intermediate_size = 1536

        weights = self._create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx=0)
        config = self._create_config(hidden_size, num_heads, num_kv_heads, intermediate_size)

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)
        x = (np.random.randn(seq_len, hidden_size) * 0.01).astype(np.float32)
        positions = np.arange(seq_len, dtype=np.int32)

        kv_cache = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=hidden_size // num_heads)
        cos_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)
        sin_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)

        out = block.forward(x, positions, kv_cache, cos_table, sin_table, is_prefill=True)

        assert out.shape == (seq_len, hidden_size), f"Expected {(seq_len, hidden_size)}, got {out.shape}"

    def test_output_shape_decode(self):
        """Output should preserve shape [1, hidden_size] during decode (single token)."""
        hidden_size = 576
        num_heads, num_kv_heads = 9, 3
        intermediate_size = 1536

        weights = self._create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx=0)
        config = self._create_config(hidden_size, num_heads, num_kv_heads, intermediate_size)

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)
        x = (np.random.randn(1, hidden_size) * 0.01).astype(np.float32)
        positions = np.array([0], dtype=np.int32)

        kv_cache = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=hidden_size // num_heads)
        cos_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)
        sin_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)

        out = block.forward(x, positions, kv_cache, cos_table, sin_table, is_prefill=False)

        assert out.shape == (1, hidden_size)

    def test_various_sequence_lengths(self):
        """Should handle various sequence lengths."""
        hidden_size = 64
        num_heads, num_kv_heads = 4, 2
        intermediate_size = 128

        weights = self._create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx=0)
        config = self._create_config(hidden_size, num_heads, num_kv_heads, intermediate_size)

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)

        cos_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)
        sin_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)

        for seq_len in [1, 5, 10, 50]:
            x = (np.random.randn(seq_len, hidden_size) * 0.01).astype(np.float32)
            positions = np.arange(seq_len, dtype=np.int32)
            kv_cache = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=hidden_size // num_heads)

            out = block.forward(x, positions, kv_cache, cos_table, sin_table, is_prefill=True)
            assert out.shape == (seq_len, hidden_size), f"Failed for seq_len={seq_len}"

    @staticmethod
    def _create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx):
        """Create fake weights for testing."""
        head_dim = hidden_size // num_heads
        return {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": (np.random.randn(num_heads * head_dim, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": (np.random.randn(num_kv_heads * head_dim, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": (np.random.randn(num_kv_heads * head_dim, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": (np.random.randn(hidden_size, num_heads * head_dim) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": (np.random.randn(intermediate_size, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.up_proj.weight": (np.random.randn(intermediate_size, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.down_proj.weight": (np.random.randn(hidden_size, intermediate_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.input_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
        }

    @staticmethod
    def _create_config(hidden_size, num_heads, num_kv_heads, intermediate_size):
        """Create config dict for testing."""
        return {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
        }


class TestLlamaDecoderBlockNumerical:
    """Test numerical properties of decoder block."""

    def test_no_nan_or_inf(self):
        """Output should not contain NaN or Inf."""
        hidden_size = 64
        num_heads, num_kv_heads = 4, 2
        intermediate_size = 128
        head_dim = hidden_size // num_heads
        seq_len = 5

        weights = {
            f"model.layers.0.self_attn.q_proj.weight": np.eye(num_heads * head_dim, hidden_size, dtype=np.float32) * 0.01,
            f"model.layers.0.self_attn.k_proj.weight": np.eye(num_kv_heads * head_dim, hidden_size, dtype=np.float32) * 0.01,
            f"model.layers.0.self_attn.v_proj.weight": np.eye(num_kv_heads * head_dim, hidden_size, dtype=np.float32) * 0.01,
            f"model.layers.0.self_attn.o_proj.weight": np.eye(hidden_size, num_heads * head_dim, dtype=np.float32) * 0.01,
            f"model.layers.0.mlp.gate_proj.weight": np.eye(intermediate_size, hidden_size, dtype=np.float32) * 0.01,
            f"model.layers.0.mlp.up_proj.weight": np.eye(intermediate_size, hidden_size, dtype=np.float32) * 0.01,
            f"model.layers.0.mlp.down_proj.weight": np.eye(hidden_size, intermediate_size, dtype=np.float32) * 0.01,
            f"model.layers.0.input_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
            f"model.layers.0.post_attention_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
        }
        config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
        }

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)
        x = (np.random.randn(seq_len, hidden_size) * 0.01).astype(np.float32)
        positions = np.arange(seq_len, dtype=np.int32)

        kv_cache = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim)
        cos_table = (np.random.randn(100, head_dim // 2) * 0.01).astype(np.float32)
        sin_table = (np.random.randn(100, head_dim // 2) * 0.01).astype(np.float32)

        out = block.forward(x, positions, kv_cache, cos_table, sin_table, is_prefill=True)

        assert not np.any(np.isnan(out)), "Output contains NaN"
        assert not np.any(np.isinf(out)), "Output contains Inf"

    def test_residual_effect(self):
        """Output should differ from input (residuals add something)."""
        hidden_size = 64
        num_heads, num_kv_heads = 4, 2
        intermediate_size = 128
        seq_len = 5

        weights = self._create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx=0)
        config = self._create_config(hidden_size, num_heads, num_kv_heads, intermediate_size)

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)
        x = (np.random.randn(seq_len, hidden_size) * 0.01).astype(np.float32)
        positions = np.arange(seq_len, dtype=np.int32)

        kv_cache = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=hidden_size // num_heads)
        cos_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)
        sin_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)

        out = block.forward(x, positions, kv_cache, cos_table, sin_table, is_prefill=True)

        assert not np.allclose(out, x), "Output should differ from input (residuals should change it)"
        assert not np.allclose(out, 0), "Output should not be all zeros"

    def test_deterministic(self):
        """Same input should produce same output."""
        hidden_size = 64
        num_heads, num_kv_heads = 4, 2
        intermediate_size = 128
        seq_len = 5

        np.random.seed(42)
        weights = self._create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx=0)
        config = self._create_config(hidden_size, num_heads, num_kv_heads, intermediate_size)

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)
        x = (np.random.randn(seq_len, hidden_size) * 0.01).astype(np.float32)
        positions = np.arange(seq_len, dtype=np.int32)

        kv_cache1 = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=hidden_size // num_heads)
        kv_cache2 = KVCache(max_seq_len=100, num_layers=1, num_kv_heads=num_kv_heads, head_dim=hidden_size // num_heads)
        cos_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)
        sin_table = (np.random.randn(100, hidden_size // num_heads // 2) * 0.01).astype(np.float32)

        out1 = block.forward(x, positions, kv_cache1, cos_table, sin_table, is_prefill=True)
        out2 = block.forward(x, positions, kv_cache2, cos_table, sin_table, is_prefill=True)

        np.testing.assert_array_almost_equal(out1, out2, decimal=5)

    @staticmethod
    def _create_fake_weights(hidden_size, num_heads, num_kv_heads, intermediate_size, layer_idx):
        """Create fake weights for testing."""
        head_dim = hidden_size // num_heads
        return {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": (np.random.randn(num_heads * head_dim, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": (np.random.randn(num_kv_heads * head_dim, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": (np.random.randn(num_kv_heads * head_dim, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": (np.random.randn(hidden_size, num_heads * head_dim) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": (np.random.randn(intermediate_size, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.up_proj.weight": (np.random.randn(intermediate_size, hidden_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.down_proj.weight": (np.random.randn(hidden_size, intermediate_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.input_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
        }

    @staticmethod
    def _create_config(hidden_size, num_heads, num_kv_heads, intermediate_size):
        """Create config dict for testing."""
        return {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
        }


class TestLlamaModelShapes:
    """Test LlamaModel forward pass shapes."""

    def test_output_shape(self):
        """Output should be [seq_len, vocab_size]."""
        seq_len = 10
        vocab_size = 1000
        hidden_size = 576
        num_layers = 2
        num_heads, num_kv_heads = 9, 3
        intermediate_size = 1536
        max_seq_len = 100

        weights, config = self._create_fake_model_weights(
            vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size, max_seq_len
        )

        class FakeSafetensors:
            def __init__(self, weights, config):
                self.tensor_data = weights
                self.config = config

        fake_st = FakeSafetensors(weights, config)
        model = LlamaModel(fake_st)

        input_ids = np.random.randint(0, vocab_size, size=seq_len).astype(np.int32)

        logits = model.forward(input_ids, kv_cache=None, is_prefill=True)

        assert logits.shape == (seq_len, vocab_size), f"Expected {(seq_len, vocab_size)}, got {logits.shape}"

    def test_single_token(self):
        """Should handle single token [1]."""
        vocab_size = 100
        hidden_size = 64
        num_layers = 1
        num_heads, num_kv_heads = 4, 2
        intermediate_size = 128
        max_seq_len = 100

        weights, config = self._create_fake_model_weights(
            vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size, max_seq_len
        )

        class FakeSafetensors:
            def __init__(self, weights, config):
                self.tensor_data = weights
                self.config = config

        fake_st = FakeSafetensors(weights, config)
        model = LlamaModel(fake_st)

        input_ids = np.array([42], dtype=np.int32)

        logits = model.forward(input_ids, kv_cache=None, is_prefill=False)

        assert logits.shape == (1, vocab_size)

    def test_no_nan_or_inf(self):
        """Output logits should not contain NaN or Inf."""
        seq_len = 5
        vocab_size = 100
        hidden_size = 64
        num_layers = 2
        num_heads, num_kv_heads = 4, 2
        intermediate_size = 128
        max_seq_len = 100

        # Use identity-like weights scaled very small to avoid numerical explosion
        head_dim = hidden_size // num_heads
        weights = {
            "model.embed_tokens.weight": np.eye(vocab_size, hidden_size, dtype=np.float32) * 0.01,
            "model.norm.weight": np.ones(hidden_size, dtype=np.float32),
        }

        for layer_idx in range(num_layers):
            # Small identity-like matrices
            weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = np.eye(num_heads * head_dim, hidden_size, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = np.eye(num_kv_heads * head_dim, hidden_size, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = np.eye(num_kv_heads * head_dim, hidden_size, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = np.eye(hidden_size, num_heads * head_dim, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = np.eye(intermediate_size, hidden_size, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = np.eye(intermediate_size, hidden_size, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = np.eye(hidden_size, intermediate_size, dtype=np.float32) * 0.01
            weights[f"model.layers.{layer_idx}.input_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)

        config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_seq_len,
            "rope_theta": 10000.0,
        }

        class FakeSafetensors:
            def __init__(self, weights, config):
                self.tensor_data = weights
                self.config = config

        fake_st = FakeSafetensors(weights, config)
        model = LlamaModel(fake_st)

        input_ids = np.random.randint(0, vocab_size, size=seq_len).astype(np.int32)

        logits = model.forward(input_ids, kv_cache=None, is_prefill=True)

        assert not np.any(np.isnan(logits)), "Logits contain NaN"
        assert not np.any(np.isinf(logits)), "Logits contain Inf"

    @staticmethod
    def _create_fake_model_weights(vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size, max_seq_len):
        """Create fake weights for full model."""
        head_dim = hidden_size // num_heads
        weights = {
            "model.embed_tokens.weight": (np.random.randn(vocab_size, hidden_size) * 0.01).astype(np.float32),
            "model.norm.weight": np.ones(hidden_size, dtype=np.float32),
        }

        for layer_idx in range(num_layers):
            weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (np.random.randn(num_heads * head_dim, hidden_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (np.random.randn(num_kv_heads * head_dim, hidden_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (np.random.randn(num_kv_heads * head_dim, hidden_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = (np.random.randn(hidden_size, num_heads * head_dim) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (np.random.randn(intermediate_size, hidden_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (np.random.randn(intermediate_size, hidden_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = (np.random.randn(hidden_size, intermediate_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.input_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)

        config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_seq_len,
            "rope_theta": 10000.0,
        }

        return weights, config


class TestSmolLM2Config:
    """Test with SmolLM2-135M specific configuration."""

    def test_decoder_block_smollm2(self):
        """Test decoder block with real SmolLM2 dimensions."""
        d_model = 576
        num_heads = 9
        num_kv_heads = 3
        head_dim = 64
        intermediate_size = 1536
        seq_len = 128

        weights = self._create_smollm2_weights(layer_idx=0)
        config = self._create_smollm2_config()

        block = LlamaDecoderBlock(weights, layer_idx=0, config=config)
        x = (np.random.randn(seq_len, d_model) * 0.001).astype(np.float32)
        positions = np.arange(seq_len, dtype=np.int32)

        kv_cache = KVCache(max_seq_len=2048, num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim)
        cos_table = (np.random.randn(2048, head_dim // 2) * 0.001).astype(np.float32)
        sin_table = (np.random.randn(2048, head_dim // 2) * 0.001).astype(np.float32)

        out = block.forward(x, positions, kv_cache, cos_table, sin_table, is_prefill=True)

        assert out.shape == (seq_len, d_model)

    def test_model_smollm2(self):
        """Test full model with SmolLM2 dimensions."""
        d_model = 576
        num_heads = 9
        num_kv_heads = 3
        head_dim = 64
        intermediate_size = 1536
        num_layers = 2
        vocab_size = 49152
        seq_len = 10

        weights, config = self._create_smollm2_model_weights(num_layers)

        class FakeSafetensors:
            def __init__(self, weights, config):
                self.tensor_data = weights
                self.config = config

        fake_st = FakeSafetensors(weights, config)
        model = LlamaModel(fake_st)

        input_ids = np.random.randint(0, vocab_size, size=seq_len).astype(np.int32)
        logits = model.forward(input_ids, kv_cache=None, is_prefill=True)

        assert logits.shape == (seq_len, vocab_size)

    @staticmethod
    def _create_smollm2_weights(layer_idx):
        """Create SmolLM2-shaped weights."""
        d_model = 576
        num_heads = 9
        num_kv_heads = 3
        head_dim = 64
        intermediate_size = 1536

        return {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": (np.random.randn(num_heads * head_dim, d_model) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": (np.random.randn(num_kv_heads * head_dim, d_model) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": (np.random.randn(num_kv_heads * head_dim, d_model) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": (np.random.randn(d_model, num_heads * head_dim) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": (np.random.randn(intermediate_size, d_model) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.up_proj.weight": (np.random.randn(intermediate_size, d_model) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.mlp.down_proj.weight": (np.random.randn(d_model, intermediate_size) * 0.01).astype(np.float32),
            f"model.layers.{layer_idx}.input_layernorm.weight": np.ones(d_model, dtype=np.float32),
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": np.ones(d_model, dtype=np.float32),
        }

    @staticmethod
    def _create_smollm2_config():
        """Create SmolLM2 config."""
        return {
            "hidden_size": 576,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "intermediate_size": 1536,
        }

    @staticmethod
    def _create_smollm2_model_weights(num_layers):
        """Create full SmolLM2 model weights."""
        d_model = 576
        num_heads = 9
        num_kv_heads = 3
        head_dim = 64
        intermediate_size = 1536
        vocab_size = 49152
        max_seq_len = 2048

        weights = {
            "model.embed_tokens.weight": (np.random.randn(vocab_size, d_model) * 0.01).astype(np.float32),
            "model.norm.weight": np.ones(d_model, dtype=np.float32),
        }

        for layer_idx in range(num_layers):
            weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (np.random.randn(num_heads * head_dim, d_model) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (np.random.randn(num_kv_heads * head_dim, d_model) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (np.random.randn(num_kv_heads * head_dim, d_model) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = (np.random.randn(d_model, num_heads * head_dim) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (np.random.randn(intermediate_size, d_model) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (np.random.randn(intermediate_size, d_model) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = (np.random.randn(d_model, intermediate_size) * 0.01).astype(np.float32)
            weights[f"model.layers.{layer_idx}.input_layernorm.weight"] = np.ones(d_model, dtype=np.float32)
            weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = np.ones(d_model, dtype=np.float32)

        config = {
            "hidden_size": d_model,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_seq_len,
            "rope_theta": 10000.0,
        }

        return weights, config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
