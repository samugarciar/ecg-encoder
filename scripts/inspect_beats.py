"""Muestra un resumen de cada columna de C1 (data/processed/beats.npz) para revisarlo rápido.

Solo lee train y val: el split de test está bloqueado durante el desarrollo (regla 4).
Uso: python scripts/inspect_beats.py
"""

from __future__ import annotations

import json

import numpy as np

from ecg.io import load_beats, validate_beats
from ecg.paths import PROCESSED_DIR


def describe(name: str, values: np.ndarray) -> str:
    """Resume una columna: forma, tipo y sus valores (conteos, rango o primeros elementos)."""
    summary = f"{name:<11} forma={values.shape}  tipo={values.dtype}"
    if values.dtype.kind in {"U", "S"}:
        uniques, counts = np.unique(values, return_counts=True)
        return f"{summary}  conteos={dict(zip(uniques.tolist(), counts.tolist()))}"
    if values.dtype == np.bool_:
        return f"{summary}  True={int(values.sum())} de {values.size}"
    if values.dtype.kind == "f":
        finite = values[np.isfinite(values)]
        summary += f"  min={finite.min():.4g}  max={finite.max():.4g}"
        if finite.size < values.size:
            summary += f"  NaN={values.size - finite.size}"
        first = [round(float(v), 4) for v in values[:5]] if values.ndim == 1 else None
    else:
        summary += f"  min={values.min()}  max={values.max()}"
        first = values[:5].tolist()
    if first is not None:
        summary += f"  primeros={first}"
    return summary


def main() -> None:
    """Carga C1 como lo harán Modelo y Evaluación, lo valida e imprime el resumen."""
    beats = load_beats(PROCESSED_DIR / "beats.npz")  # train y val por defecto
    validate_beats(beats)
    print(f"C1 · {len(beats['X']):,} latidos de train y val\n")
    for name, values in beats.items():
        print(describe(name, values))

    meta_path = PROCESSED_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"\nmeta.json · data_version {meta['data_version']} · "
              f"commit {meta['git_commit'][:7]} · creado {meta['created_at']}")


if __name__ == "__main__":
    main()
