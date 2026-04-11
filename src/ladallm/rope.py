"""RoPE (Rotary Position Embedding) implementation for LadaLLM."""

from typing import Tuple

import numpy as np


def precompute_rope_tables(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cosine and sine tables.

    Computes rotation angles theta_{m,i} = m * base^(-2i/head_dim)
    for all positions m in [0, max_seq_len) and dimension pairs i.

    Args:
        max_seq_len: Maximum sequence length (e.g., 2048 for SmolLM2)
        head_dim: Dimension of each head (e.g., 64)
        base: Base frequency for angle calculation (default 10000.0)

    Returns:
        cos_table: [max_seq_len, head_dim/2] - Cosine of rotation angles
        sin_table: [max_seq_len, head_dim/2] - Sine of rotation angles
    """
    angles = np.arange(max_seq_len).reshape(-1, 1) / (
        base ** (np.arange(0, head_dim, 2) / head_dim)
    )
    return np.cos(angles), np.sin(angles)


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    positions: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Position Embedding to query and key tensors.

    Rotates pairs of dimensions in Q and K using precomputed tables.
    Each pair [x, y] is rotated by angle theta to [x*cos - y*sin, x*sin + y*cos].

    Args:
        q: Query tensor [seq_len, num_heads, head_dim]
        k: Key tensor [seq_len, num_kv_heads, head_dim]
        positions: Positional indices for each token [seq_len]
        cos_table: Precomputed cosine table [max_seq_len, head_dim/2]
        sin_table: Precomputed sine table [max_seq_len, head_dim/2]

    Returns:
        q_rotated: Rotated query [seq_len, num_heads, head_dim]
        k_rotated: Rotated key [seq_len, num_kv_heads, head_dim]
    """
    cos = cos_table[positions][:, None, :]  # [seq_len, 1, head_dim/2]
    sin = sin_table[positions][:, None, :]  # [seq_len, 1, head_dim/2]

    x_q = q[..., 0::2]  # [seq_len, num_heads, head_dim/2]
    y_q = q[..., 1::2]  # [seq_len, num_heads, head_dim/2]
    x_k = k[..., 0::2]  # [seq_len, num_kv_heads, head_dim/2]
    y_k = k[..., 1::2]  # [seq_len, num_kv_heads, head_dim/2]

    q_rotated = np.empty_like(q)  # [seq_len, num_heads, head_dim]
    k_rotated = np.empty_like(k)  # [seq_len, num_kv_heads, head_dim]

    q_rotated[..., 0::2] = x_q * cos - y_q * sin
    q_rotated[..., 1::2] = x_q * sin + y_q * cos
    k_rotated[..., 0::2] = x_k * cos - y_k * sin
    k_rotated[..., 1::2] = x_k * sin + y_k * cos

    return q_rotated, k_rotated
