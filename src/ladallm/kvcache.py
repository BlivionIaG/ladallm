"""KV cache for LadaLLM."""

from typing import Tuple

import numpy as np


class KVCache:
    """Pre-allocated KV cache for efficient attention.

    Stores K and V tensors for all layers and positions to avoid
    recomputing them during autoregressive generation.

    Memory layout: [num_layers, max_seq_len, num_kv_heads, head_dim]
    This enables sequential access during decode (cache-friendly).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=np.float32,
    ):
        """Initialize KV cache.

        Args:
            max_seq_len: Maximum sequence length to support
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            dtype: Data type for cache tensors
        """
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # [num_layers, max_seq_len, num_kv_heads, head_dim] for sequential access
        self.k = np.zeros((num_layers, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        self.v = np.zeros((num_layers, max_seq_len, num_kv_heads, head_dim), dtype=dtype)

        self.length = 0
        self.current_layer = 0

    def append(self, k: np.ndarray, v: np.ndarray):
        """Append K,V for current tokens.

        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
        """
        seq_len = k.shape[0]
        self.k[self.current_layer, self.length : self.length + seq_len] = k
        self.v[self.current_layer, self.length : self.length + seq_len] = v
        self.length += seq_len

    def __getitem__(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached K,V up to current length.

        Args:
            layer: Layer index

        Returns:
            k: Cached keys [length, num_kv_heads, head_dim]
            v: Cached values [length, num_kv_heads, head_dim]
        """
        return self.k[layer, : self.length], self.v[layer, : self.length]

    def __len__(self) -> int:
        """Return current cache length."""
        return self.length

    def next_layer(self):
        """Move to next layer (for multi-layer processing)."""
        self.current_layer += 1
