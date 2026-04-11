"""Llama model implementation for LadaLLM."""

import numpy as np

from ladallm.attention import attention_forward, causal_mask, compute_qkv
from ladallm.kvcache import KVCache
from ladallm.rope import apply_rope, precompute_rope_tables
from ladallm.safetensors import Safetensors


class LlamaModel:
    """Llama architecture model with multi-layer transformer.

    This class owns the complete model state including weights,
    configuration, and precomputed RoPE tables.

    Attributes:
        config: Model configuration dictionary
        weights: Tensor dictionary from safetensors
        hidden_size: Hidden dimension (e.g., 576 for SmolLM2)
        num_heads: Number of attention heads (e.g., 9 for SmolLM2)
        num_kv_heads: Number of KV heads for GQA (e.g., 3 for SmolLM2)
        head_dim: Dimension per head (e.g., 64 for SmolLM2)
        num_layers: Number of transformer layers (e.g., 30 for SmolLM2)
        vocab_size: Vocabulary size (e.g., 49152 for SmolLM2)
        cos_table: Precomputed RoPE cosine table [max_seq_len, head_dim/2]
        sin_table: Precomputed RoPE sine table [max_seq_len, head_dim/2]
        layers: List of ModelLayer instances
    """

    def __init__(self, safetensors: Safetensors):
        """Initialize Llama model from loaded safetensors.

        Args:
            safetensors: Loaded Safetensors instance containing weights and config
        """
        self.config = safetensors.config
        self.weights = safetensors.tensor_data

        # Extract architecture parameters from config
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_layers = self.config["num_hidden_layers"]
        self.vocab_size = self.config["vocab_size"]
        self.max_seq_len = self.config["max_position_embeddings"]
        self.rope_theta = self.config["rope_theta"]

        # Precompute RoPE tables once (shared across all layers)
        self.cos_table, self.sin_table = precompute_rope_tables(
            max_seq_len=self.max_seq_len,
            head_dim=self.head_dim,
            base=self.rope_theta,
        )

        # Create all transformer layers
        self.layers = [
            ModelLayer(
                weights=self.weights,
                layer_idx=i,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
            )
            for i in range(self.num_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        positions: np.ndarray,
        kv_cache: KVCache = None,
        is_prefill: bool = True,
    ) -> np.ndarray:
        """Run forward pass through all layers.

        Args:
            x: Input tensor [seq_len, hidden_size]
            positions: Position indices [seq_len]
            kv_cache: Optional KV cache for storing/retrieving K,V
            is_prefill: True for prefill phase, False for decode

        Returns:
            out: Output tensor [seq_len, hidden_size]
        """
        for layer in self.layers:
            x = layer.forward(
                x=x,
                positions=positions,
                kv_cache=kv_cache,
                cos_table=self.cos_table,
                sin_table=self.sin_table,
                is_prefill=is_prefill,
            )
        return x


class ModelLayer:
    """Single transformer layer with attention.

    This represents one decoder block consisting of:
    1. RMSNorm (to be added in F6)
    2. Self-attention with RoPE and KV cache
    3. Output projection
    4. Residual connection (to be added in F6)

    Attributes:
        w_q: Query projection weight [hidden_size, num_heads*head_dim]
        w_k: Key projection weight [hidden_size, num_kv_heads*head_dim]
        w_v: Value projection weight [hidden_size, num_kv_heads*head_dim]
        w_o: Output projection weight [num_heads*head_dim, hidden_size]
        layer_idx: Layer index (0 to num_layers-1)
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA
        head_dim: Dimension per head
        hidden_size: Model hidden size
    """

    def __init__(
        self,
        weights: dict,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        """Initialize layer.

        Args:
            weights: Weight dictionary from safetensors
            layer_idx: Layer index (0 to num_layers-1)
            num_heads: Number of attention heads (e.g., 9 for SmolLM2)
            num_kv_heads: Number of KV heads for GQA (e.g., 3 for SmolLM2)
            head_dim: Dimension per head (e.g., 64 for SmolLM2)
            hidden_size: Model hidden size (e.g., 576 for SmolLM2)
        """
        # Safetensors keys: model.layers.{idx}.self_attn.{q,k,v,o}_proj.weight
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.w_q = weights[f"{prefix}.q_proj.weight"]
        self.w_k = weights[f"{prefix}.k_proj.weight"]
        self.w_v = weights[f"{prefix}.v_proj.weight"]
        self.w_o = weights[f"{prefix}.o_proj.weight"]

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.attn_scale = 1.0 / np.sqrt(head_dim)

    def forward(
        self,
        x: np.ndarray,
        positions: np.ndarray,
        kv_cache: KVCache,
        cos_table: np.ndarray,
        sin_table: np.ndarray,
        is_prefill: bool = True,
    ) -> np.ndarray:
        """Forward pass through layer.

        Args:
            x: Input tensor [seq_len, hidden_size]
            positions: Position indices [seq_len]
            kv_cache: KV cache for storing/retrieving K,V
            cos_table: RoPE cos table [max_seq_len, head_dim/2]
            sin_table: RoPE sin table [max_seq_len, head_dim/2]
            is_prefill: True for prefill phase, False for decode

        Returns:
            out: Output tensor [seq_len, hidden_size]
        """
        seq_len = x.shape[0]

        # 1. QKV Projection
        q, k, v = compute_qkv(
            x,
            self.w_q,
            self.w_k,
            self.w_v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        # 2. Apply RoPE
        q, k = apply_rope(q, k, positions, cos_table, sin_table)

        # 3. Update cache and retrieve cached K,V
        if kv_cache is not None:
            kv_cache.append(k, v)
            k_cached, v_cached = kv_cache[self.layer_idx]
        else:
            k_cached, v_cached = k, v

        # 4. Compute causal mask for prefill
        mask = None
        if is_prefill and seq_len > 1:
            mask = causal_mask(seq_len, k_cached.shape[0])

        # 5. Attention
        attn_out = attention_forward(
            q,
            k_cached,
            v_cached,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attn_scale=self.attn_scale,
            mask=mask,
        )

        # 6. Output projection
        # attn_out: [seq_len, num_heads, head_dim]
        attn_out = attn_out.reshape(seq_len, self.num_heads * self.head_dim)
        out = attn_out @ self.w_o.T

        return out
