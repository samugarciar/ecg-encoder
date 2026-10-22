"""Calcula el error de reconstrucción de cada latido de un split (contrato C2).

Uso: python -m ecg.score --run runs/<run_id> --split val
Escribe runs/<run_id>/scores_<split>.npz. Desde aquí solo se puntúan train y val: el split de
test lo puntúa src/ecg/evaluate/final_test.py, con allow_test=True (regla 4).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg.io import load_beats
from ecg.models.autoencoder import ConvAutoencoder
from ecg.models.preprocessing import prepare_inputs
from ecg.paths import PROCESSED_DIR
from ecg.train import build_model, load_config

# Claves y tipos del contrato C2 (docs/contracts.md)
C2_DTYPES: dict[str, Any] = {
    "idx": np.int64,
    "error": np.float32,
    "label": np.int8,
    "symbol": "<U1",
    "aami": "<U1",
    "record": np.int16,
    "mlii_noisy": np.bool_,
}


def load_run(run_dir: Path | str) -> tuple[ConvAutoencoder, dict[str, Any]]:
    """Carga la configuración y los pesos de la mejor época de un run, listo para evaluar."""
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    model = build_model(config)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, config


def reconstruct(
    model: ConvAutoencoder, X: np.ndarray, normalization: str, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza los latidos igual que en el entrenamiento y devuelve (entrada, reconstrucción)."""
    inputs = prepare_inputs(X, normalization)
    outputs = np.empty_like(inputs)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            xb = torch.from_numpy(inputs[start:start + batch_size]).unsqueeze(1)
            outputs[start:start + batch_size] = model(xb).squeeze(1).numpy()
    return inputs, outputs


def reconstruction_error(
    model: ConvAutoencoder, X: np.ndarray, normalization: str, batch_size: int = 1024
) -> np.ndarray:
    """MSE por latido entre la entrada normalizada y su reconstrucción (columna `error` de C2)."""
    inputs, outputs = reconstruct(model, X, normalization, batch_size)
    return np.mean((inputs - outputs) ** 2, axis=1).astype(np.float32)


def validate_scores(scores: Mapping[str, np.ndarray]) -> bool:
    """Verifica claves, formas y tipos del contrato C2; lanza ValueError o TypeError si falla."""
    missing = set(C2_DTYPES) - set(scores)
    if missing:
        raise ValueError(f"Faltan claves del Contrato C2: {sorted(missing)}")
    m = len(scores["idx"])
    for key, dtype in C2_DTYPES.items():
        arr = np.asarray(scores[key])
        if arr.shape != (m,):
            raise ValueError(f"'{key}' debe tener forma ({m},), tiene {arr.shape}")
        if arr.dtype != np.dtype(dtype):
            raise TypeError(f"'{key}' debe ser {np.dtype(dtype)}, se encontró {arr.dtype}")
    error = np.asarray(scores["error"])
    if not np.all(np.isfinite(error)) or np.any(error < 0):
        raise ValueError("'error' debe ser finito y no negativo")
    return True


def score_split(
    run_dir: Path | str,
    split: str,
    beats_path: Path | str | None = None,
    allow_test: bool = False,
) -> dict[str, np.ndarray]:
    """Error de reconstrucción de todos los latidos de `split`, con las columnas del contrato C2.

    `idx` es la fila de cada latido en el beats.npz completo. Se puntúan también los latidos
    ruidosos: `mlii_noisy` permite a Evaluación separarlos (D-020).
    """
    beats_path = Path(beats_path) if beats_path is not None else PROCESSED_DIR / "beats.npz"
    beats = load_beats(beats_path, splits=(split,), allow_test=allow_test)
    with np.load(beats_path, allow_pickle=False) as data:
        idx = np.flatnonzero(np.asarray(data["split"]).astype(str) == split)
    if len(idx) != len(beats["X"]):
        raise ValueError("Los índices de fila no coinciden con los latidos cargados del split")

    model, config = load_run(run_dir)
    error = reconstruction_error(model, beats["X"], config["data"]["normalization"])
    scores = {
        "idx": idx,
        "error": error,
        "label": beats["label"],
        "symbol": beats["symbol"],
        "aami": beats["aami"],
        "record": beats["record"],
        "mlii_noisy": beats["mlii_noisy"],
    }
    scores = {key: np.asarray(value).astype(C2_DTYPES[key]) for key, value in scores.items()}
    validate_scores(scores)
    return scores


def save_scores(run_dir: Path | str, split: str, scores: Mapping[str, np.ndarray]) -> Path:
    """Guarda las puntuaciones en runs/<run_id>/scores_<split>.npz."""
    validate_scores(scores)
    path = Path(run_dir) / f"scores_{split}.npz"
    np.savez_compressed(path, **scores)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula el contrato C2 de un run.")
    parser.add_argument("--run", required=True, help="carpeta runs/<run_id>")
    parser.add_argument("--split", required=True, choices=["train", "val"])
    args = parser.parse_args()
    scores = score_split(args.run, args.split)
    path = save_scores(args.run, args.split, scores)
    median = float(np.median(scores["error"]))
    print(f"{path}: {len(scores['idx']):,} latidos · error mediano {median:.4f}")


if __name__ == "__main__":
    main()
