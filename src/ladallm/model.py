"""Llama model implementation for LadaLLM."""

import numpy as np

from ladallm.attention import attention_forward, causal_mask, compute_qkv
from ladallm.cli import rms_norm
from ladallm.kvcache import KVCache
from ladallm.mlp import swiglu_mlp
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
        layers: List of LlamaDecoderBlock instances
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

        # Load embedding and output weights
        self.embed_tokens = self.weights["model.embed_tokens.weight"]
        self.norm_weight = self.weights["model.norm.weight"]
        # LM head is tied to embeddings for SmolLM2
        self.lm_head = self.embed_tokens

        # Create all transformer layers
        self.layers = [
            LlamaDecoderBlock(
                weights=self.weights,
                layer_idx=i,
                config=self.config,
            )
            for i in range(self.num_layers)
        ]

    def forward(
        self,
        input_ids: np.ndarray,
        kv_cache: KVCache = None,
        is_prefill: bool = True,
    ) -> np.ndarray:
        """Run forward pass through all layers.

        Args:
            input_ids: Input token IDs [seq_len]
            kv_cache: Optional KV cache for storing/retrieving K,V
            is_prefill: True for prefill phase, False for decode

        Returns:
            logits: Output logits [seq_len, vocab_size]
        """
        # 1. Embedding lookup
        x = self.embed_tokens[input_ids]  # [seq_len, hidden_size]

        # 2. Position indices for RoPE
        positions = np.arange(len(input_ids), dtype=np.int32)

        # 3. Pass through all decoder blocks
        for layer in self.layers:
            x = layer.forward(
                x=x,
                positions=positions,
                kv_cache=kv_cache,
                cos_table=self.cos_table,
                sin_table=self.sin_table,
                is_prefill=is_prefill,
            )

        # 4. Final RMSNorm
        x = rms_norm(x, self.norm_weight)

        # 5. LM head projection to vocabulary
        logits = x @ self.lm_head.T  # [seq_len, vocab_size]

        return logits


class LlamaDecoderBlock:
    """Single transformer decoder block: Norm→Attn→Resid→Norm→MLP→Resid.

    This represents one decoder block consisting of:
    1. RMSNorm before attention (input_layernorm)
    2. Self-attention with RoPE and KV cache
    3. Residual connection
    4. RMSNorm before MLP (post_attention_layernorm)
    5. SwiGLU MLP
    6. Residual connection

    Attributes:
        w_q: Query projection weight [hidden_size, num_heads*head_dim]
        w_k: Key projection weight [hidden_size, num_kv_heads*head_dim]
        w_v: Value projection weight [hidden_size, num_kv_heads*head_dim]
        w_o: Output projection weight [num_heads*head_dim, hidden_size]
        w_gate: MLP gate projection weight [hidden_size, intermediate_size]
        w_up: MLP up projection weight [hidden_size, intermediate_size]
        w_down: MLP down projection weight [intermediate_size, hidden_size]
        norm_attn: Attention pre-norm weight [hidden_size]
        norm_mlp: MLP pre-norm weight [hidden_size]
        layer_idx: Layer index (0 to num_layers-1)
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA
        head_dim: Dimension per head
        hidden_size: Model hidden size
        intermediate_size: MLP intermediate dimension
        attn_scale: Precomputed attention scale (1/sqrt(head_dim))
    """

    def __init__(
        self,
        weights: dict,
        layer_idx: int,
        config: dict,
    ):
        """Initialize decoder block.

        Args:
            weights: Weight dictionary from safetensors
            layer_idx: Layer index (0 to num_layers-1)
            config: Model configuration dictionary containing:
                - hidden_size: Model hidden dimension
                - num_attention_heads: Number of query heads
                - num_key_value_heads: Number of KV heads (for GQA)
                - intermediate_size: MLP intermediate dimension
        """
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config["intermediate_size"]
        self.attn_scale = 1.0 / np.sqrt(self.head_dim)

        # Load attention weights
        attn_prefix = f"model.layers.{layer_idx}.self_attn"
        self.w_q = weights[f"{attn_prefix}.q_proj.weight"]
        self.w_k = weights[f"{attn_prefix}.k_proj.weight"]
        self.w_v = weights[f"{attn_prefix}.v_proj.weight"]
        self.w_o = weights[f"{attn_prefix}.o_proj.weight"]

        # Load MLP weights (SwiGLU)
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        self.w_gate = weights[f"{mlp_prefix}.gate_proj.weight"]
        self.w_up = weights[f"{mlp_prefix}.up_proj.weight"]
        self.w_down = weights[f"{mlp_prefix}.down_proj.weight"]

        # Load RMSNorm weights
        self.norm_attn = weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
        self.norm_mlp = weights[
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        ]

    def forward(
        self,
        x: np.ndarray,
        positions: np.ndarray,
        kv_cache: KVCache,
        cos_table: np.ndarray,
        sin_table: np.ndarray,
        is_prefill: bool = True,
    ) -> np.ndarray:
        """Forward pass through decoder block.

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

        # === ATTENTION SUBLAYER ===
        # 1. Pre-normalization
        normed = rms_norm(x, self.norm_attn)

        # 2. QKV projection
        q, k, v = compute_qkv(
            normed,
            self.w_q,
            self.w_k,
            self.w_v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        # 3. Apply RoPE
        q, k = apply_rope(q, k, positions, cos_table, sin_table)

        # 4. Update cache and retrieve cached K,V
        if kv_cache is not None:
            kv_cache.append(k, v)
            k_cached, v_cached = kv_cache[self.layer_idx]
        else:
            k_cached, v_cached = k, v

        # 5. Compute causal mask for prefill
        mask = None
        if is_prefill and seq_len > 1:
            mask = causal_mask(seq_len, k_cached.shape[0])

        # 6. Attention
        attn_out = attention_forward(
            q,
            k_cached,
            v_cached,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attn_scale=self.attn_scale,
            mask=mask,
        )

        # 7. Output projection
        attn_out = attn_out.reshape(seq_len, self.num_heads * self.head_dim)
        attn_out = attn_out @ self.w_o.T

        # 8. Residual connection
        x += attn_out

        # === MLP SUBLAYER ===
        # 9. Pre-normalization
        normed = rms_norm(x, self.norm_mlp)

        # 10. SwiGLU MLP
        mlp_out = swiglu_mlp(normed, self.w_gate.T, self.w_up.T, self.w_down.T)

        # 11. Residual connection
        x += mlp_out

        return x
