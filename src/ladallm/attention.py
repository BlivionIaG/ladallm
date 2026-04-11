"""Attention mechanism for LadaLLM."""

from typing import Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_qkv(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project input to Q, K, V.

    Args:
        x: Input tensor [seq_len, hidden_size]
        w_q: Query weight [num_heads*head_dim, hidden_size]
        w_k: Key weight [num_kv_heads*head_dim, hidden_size]
        w_v: Value weight [num_kv_heads*head_dim, hidden_size]
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head

    Returns:
        q: Query tensor [seq_len, num_heads, head_dim]
        k: Key tensor [seq_len, num_kv_heads, head_dim]
        v: Value tensor [seq_len, num_kv_heads, head_dim]
    """
    seq_len = x.shape[0]
    return (
        (x @ w_q.T).reshape(seq_len, num_heads, head_dim),
        (x @ w_k.T).reshape(seq_len, num_kv_heads, head_dim),
        (x @ w_v.T).reshape(seq_len, num_kv_heads, head_dim),
    )


def causal_mask(seq_len_q: int, seq_len_k: int) -> np.ndarray:
    """Create causal mask with 0 for allowed, -inf for forbidden.

    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length

    Returns:
        mask: [seq_len_q, seq_len_k] with 0 for allowed, -inf for forbidden
    """
    return np.triu(np.ones((seq_len_q, seq_len_k)), k=1) * float("-inf")


def attention_forward(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    head_dim: int,
    num_kv_heads: int,
    num_heads: int,
    attn_scale: float,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Compute attention.

    Args:
        q: Query tensor [num_heads, head_dim] (decode) or [seq_len, num_heads, head_dim] (prefill)
        k_cache: Cached keys [cache_len, num_kv_heads, head_dim]
        v_cache: Cached values [cache_len, num_kv_heads, head_dim]
        head_dim: Dimension per head
        num_kv_heads: Number of KV heads
        num_heads: Number of attention heads
        attn_scale: Precomputed attention scale (1/sqrt(head_dim))
        mask: Optional causal mask [seq_q, seq_k]

    Returns:
        out: Attention output [seq_q, num_heads, head_dim]
    """
    group_size = num_heads // num_kv_heads
    k = np.repeat(k_cache, group_size, axis=1)
    v = np.repeat(v_cache, group_size, axis=1)

    if q.ndim == 2:
        q = q[np.newaxis, ...]

    scores = np.einsum("qhd,khd->qkh", q, k) * attn_scale

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    return np.einsum("qkh,khd->qhd", weights, v)
