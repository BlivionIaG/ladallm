"""KV cache for LadaLLM.

This module provides the NaiveKVCache class for storing Key and Value tensors
during autoregressive generation. Each transformer layer has its own cache
instance, enabling efficient memory management and append-only access patterns.

The naive implementation pre-allocates contiguous buffers to max_seq_len,
which wastes memory for short sequences but provides predictable performance
and simple indexing. This serves as the v0 baseline before paged attention
(v1) introduces block-based allocation.

Example:
    >>> cache = NaiveKVCache(max_seq_len=2048, num_kv_heads=3, head_dim=64)
    >>> # During prefill: append multiple tokens at once
    >>> k_prefill = np.random.randn(10, 3, 64)  # 10 prompt tokens
    >>> v_prefill = np.random.randn(10, 3, 64)
    >>> cache.append(k_prefill, v_prefill)
    >>> # During decode: append one token at a time
    >>> k_decode = np.random.randn(1, 3, 64)  # 1 new token
    >>> v_decode = np.random.randn(1, 3, 64)
    >>> cache.append(k_decode, v_decode)
    >>> # Retrieve all cached K, V for attention
    >>> k_cached, v_cached = cache.get()
    >>> print(k_cached.shape)  # (11, 3, 64) - 10 prefill + 1 decode
"""

from typing import Tuple

import numpy as np


class NaiveKVCache:
    """Pre-allocated KV cache for a single transformer layer.

    Stores K and V tensors for all positions in a single layer to avoid
    recomputing them during autoregressive generation. This is the "naive"
    contiguous allocation approach - simple, predictable, but potentially
    wasteful of memory for short sequences.

    Memory layout: [max_seq_len, num_kv_heads, head_dim]
    - Dimension 0 (seq_len): Pre-allocated to max_seq_len, grows dynamically
    - Dimension 1 (num_kv_heads): Separate cache per KV head (GQA support)
    - Dimension 2 (head_dim): Per-head feature dimension

    The cache grows monotonically via append() during generation:
    1. Prefill: Append K/V for all prompt tokens at once
    2. Decode: Append K/V for one new token per step

    Attributes:
        k: Key tensor buffer [max_seq_len, num_kv_heads, head_dim]
        v: Value tensor buffer [max_seq_len, num_kv_heads, head_dim]
        length: Current number of cached positions (0 to max_seq_len)
        max_seq_len: Maximum sequence length this cache can hold
        num_kv_heads: Number of KV heads (for Grouped Query Attention)
        head_dim: Dimension per head

    Example:
        >>> cache = NaiveKVCache(max_seq_len=100, num_kv_heads=4, head_dim=64)
        >>>
        >>> # Prefill with 5 tokens
        >>> k = np.random.randn(5, 4, 64)
        >>> v = np.random.randn(5, 4, 64)
        >>> cache.append(k, v)
        >>> assert len(cache) == 5
        >>>
        >>> # Decode: add tokens one at a time
        >>> for i in range(3):
        ...     k_new = np.random.randn(1, 4, 64)
        ...     v_new = np.random.randn(1, 4, 64)
        ...     cache.append(k_new, v_new)
        >>> assert len(cache) == 8
        >>>
        >>> # Get all cached K, V for attention computation
        >>> k_all, v_all = cache.get()
        >>> print(k_all.shape)  # (8, 4, 64)
    """

    def __init__(
        self,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ):
        """Initialize a single layer's KV cache.

        Allocates contiguous buffers to max_seq_len. The actual usage
        starts at 0 and grows via append() during generation.

        Args:
            max_seq_len: Maximum sequence length this cache can support.
                Determines the allocation size for k and v buffers.
            num_kv_heads: Number of KV heads for Grouped Query Attention.
                This is the compressed head count (e.g., 3 for SmolLM2),
                not the full query head count.
            head_dim: Dimension per head (hidden_size // num_heads).
                For SmolLM2: 576 // 9 = 64.
            dtype: NumPy data type for cache tensors. Default float32.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocate contiguous buffers
        # Shape: [max_seq_len, num_kv_heads, head_dim]
        # Access pattern during decode: sequential read of [:length, :, :]
        self.k: np.ndarray = np.zeros(
            (max_seq_len, num_kv_heads, head_dim), dtype=dtype
        )
        self.v: np.ndarray = np.zeros(
            (max_seq_len, num_kv_heads, head_dim), dtype=dtype
        )

        # Current number of valid positions in the cache
        # Starts at 0, grows monotonically during generation
        self._length: int = 0

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        """Append new K,V tensors to the cache.

        Writes k_new and v_new at positions [length, length + seq_len)
        and increments length. This is called:
        - Once during prefill with seq_len = prompt_length
        - Once per decode step with seq_len = 1

        Args:
            k_new: Key tensor to append. Shape: [seq_len, num_kv_heads, head_dim]
            v_new: Value tensor to append. Shape: [seq_len, num_kv_heads, head_dim]

        Raises:
            ValueError: If shapes don't match cache dimensions.
            RuntimeError: If appending would exceed max_seq_len.

        Example:
            >>> cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)
            >>> # Prefill: 10 tokens at once
            >>> k = np.random.randn(10, 3, 64).astype(np.float32)
            >>> v = np.random.randn(10, 3, 64).astype(np.float32)
            >>> cache.append(k, v)
            >>> assert len(cache) == 10
            >>> # Decode: 1 token at a time
            >>> k_new = np.random.randn(1, 3, 64).astype(np.float32)
            >>> v_new = np.random.randn(1, 3, 64).astype(np.float32)
            >>> cache.append(k_new, v_new)
            >>> assert len(cache) == 11
        """
        if k_new.shape != v_new.shape:
            raise ValueError(
                f"k_new and v_new shapes must match: {k_new.shape} vs {v_new.shape}"
            )

        expected_shape_suffix = (self.num_kv_heads, self.head_dim)
        if k_new.shape[1:] != expected_shape_suffix:
            raise ValueError(
                f"Expected K,V shape (..., {expected_shape_suffix}), got {k_new.shape}"
            )

        seq_len = k_new.shape[0]

        if self._length + seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Cache overflow: cannot append {seq_len} tokens "
                f"(current length: {self._length}, max: {self.max_seq_len})"
            )

        # Write to the next available positions
        end_pos = self._length + seq_len
        self.k[self._length : end_pos] = k_new
        self.v[self._length : end_pos] = v_new
        self._length = end_pos

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached K,V tensors up to current length.

        Returns views into the pre-allocated buffers, not copies.
        This is used by attention_forward to read all past K,V.

        Returns:
            Tuple of (k_cached, v_cached), each with shape
            [length, num_kv_heads, head_dim] where length is the
            current number of cached positions.

        Example:
            >>> cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)
            >>> cache.append(np.random.randn(5, 3, 64), np.random.randn(5, 3, 64))
            >>> k, v = cache.get()
            >>> print(k.shape)  # (5, 3, 64)
            >>> print(v.shape)  # (5, 3, 64)
        """
        return self.k[: self._length], self.v[: self._length]

    def __len__(self) -> int:
        """Return the number of cached positions.

        This starts at 0 after initialization and grows monotonically
        as append() is called during generation.

        Returns:
            Current cache length (number of stored positions).

        Example:
            >>> cache = NaiveKVCache(max_seq_len=100, num_kv_heads=3, head_dim=64)
            >>> len(cache)  # 0
            >>> cache.append(np.random.randn(5, 3, 64), np.random.randn(5, 3, 64))
            >>> len(cache)  # 5
        """
        return self._length

    @property
    def memory_usage_bytes(self) -> int:
        """Get total memory usage of this cache in bytes.

        Returns:
            Total bytes allocated for k and v buffers (2 * max_seq_len * num_kv_heads * head_dim * dtype_size).

        Example:
            >>> cache = NaiveKVCache(max_seq_len=2048, num_kv_heads=3, head_dim=64)
            >>> print(f"Cache uses {cache.memory_usage_bytes / 1024 / 1024:.2f} MB")
        """
        return self.k.nbytes + self.v.nbytes


def create_layer_caches(
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: np.dtype = np.float32,
) -> list[NaiveKVCache]:
    """Create a list of NaiveKVCache instances, one per layer.

    This helper creates the per-layer cache structure needed by the
    LlamaModel during generation. Each layer gets its own independent
    cache instance.

    Args:
        max_seq_len: Maximum sequence length for each cache.
        num_layers: Number of transformer layers (e.g., 30 for SmolLM2).
        num_kv_heads: Number of KV heads per layer.
        head_dim: Dimension per head.
        dtype: NumPy data type for cache tensors.

    Returns:
        List of NaiveKVCache instances, one per layer.

    Example:
        >>> caches = create_layer_caches(
        ...     max_seq_len=2048,
        ...     num_layers=30,
        ...     num_kv_heads=3,
        ...     head_dim=64
        ... )
        >>> len(caches)  # 30
        >>> total_memory = sum(c.memory_usage_bytes for c in caches)
        >>> print(f"Total KV cache: {total_memory / 1024 / 1024:.2f} MB")
    """
    return [
        NaiveKVCache(max_seq_len, num_kv_heads, head_dim, dtype)
        for _ in range(num_layers)
    ]
