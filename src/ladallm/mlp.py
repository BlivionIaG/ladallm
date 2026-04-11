"""SwiGLU MLP implementation for LadaLLM."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


def swiglu_mlp(
    x: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    W_down: np.ndarray,
) -> np.ndarray:
    """SwiGLU MLP forward pass.

    Args:
        x: Input tensor [seq_len, d_model]
        W_gate: Gate projection weight [d_model, d_ff]
        W_up: Up projection weight [d_model, d_ff]
        W_down: Down projection weight [d_ff, d_model]

    Returns:
        Output tensor [seq_len, d_model]
    """
    gate = x @ W_gate
    np.multiply(gate, sigmoid(gate), out=gate)  # in-place SiLU
    np.multiply(gate, x @ W_up, out=gate)  # gate *= up projection (computed on-the-fly)
    return gate @ W_down
