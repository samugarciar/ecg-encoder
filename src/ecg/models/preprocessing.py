"""Preparación de los latidos de C1 para el autoencoder (D-008, D-020)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

NORMALIZATIONS = ("zscore", "none")


def zscore_beats(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """z-score por latido (D-008): a cada fila le resta su media y la divide por su desviación.

    `eps` evita dividir por cero si algún latido fuera plano.
    """
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return ((X - mean) / np.maximum(std, eps)).astype(np.float32)


def prepare_inputs(X: np.ndarray, normalization: str) -> np.ndarray:
    """Aplica la normalización configurada y devuelve un array float32 de forma (n, L)."""
    if normalization == "zscore":
        return zscore_beats(X)
    if normalization == "none":
        return np.asarray(X, dtype=np.float32)
    raise ValueError(
        f"normalization debe ser una de {NORMALIZATIONS}, se recibió {normalization!r}"
    )


def normal_mask(beats: Mapping[str, np.ndarray], split: str, exclude_noisy: bool) -> np.ndarray:
    """Latidos normales (`label == 0`) de un split; sin los ruidosos si `exclude_noisy` (D-020)."""
    mask = (np.asarray(beats["split"]).astype(str) == split) & (np.asarray(beats["label"]) == 0)
    if exclude_noisy:
        mask &= ~np.asarray(beats["mlii_noisy"], dtype=bool)
    return mask
