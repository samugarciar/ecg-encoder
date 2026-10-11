"""Funciones de entrada/salida y validación de contratos para latidos ECG.

Regla 4 de AGENTS.md: El split 'test' (DS2) está prohibido en desarrollo.
Solo src/ecg/evaluate/final_test.py lo usa, con --final y allow_test=True.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ecg.paths import PROCESSED_DIR

# Claves requeridas por el contrato C1
C1_REQUIRED_KEYS: frozenset[str] = frozenset({
    "X",
    "symbol",
    "aami",
    "label",
    "record",
    "split",
    "r_sample",
    "rr_prev",
    "rr_next",
    "mlii_noisy",
})

ALLOWED_AAMI: frozenset[str] = frozenset({"N", "S", "V", "F", "Q"})
ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


def load_beats(
    path: Path | str | None = None,
    splits: Sequence[str] = ("train", "val"),
    allow_test: bool = False,
) -> dict[str, np.ndarray]:
    """Carga y filtra los latidos preprocesados por subconjunto (split).

    Parámetros
    ----------
    path:
        Ruta al archivo .npz (por defecto `data/processed/beats.npz`).
    splits:
        Colección de splits a incluir ('train', 'val', 'test').
    allow_test:
        Bandera de seguridad obligatoria para poder acceder al split 'test' (DS2).

    Retorna
    -------
    dict[str, np.ndarray]:
        Diccionario con los arrays filtrados para los splits solicitados.

    Excepciones
    -----------
    PermissionError:
        Si se solicita el split 'test' y allow_test es False.
    FileNotFoundError:
        Si el archivo .npz no existe en la ruta especificada.
    """
    splits_set = set(splits)
    if "test" in splits_set and not allow_test:
        raise PermissionError(
            "Acceso denegado: El split 'test' (DS2) está prohibido durante la fase de desarrollo. "
            "Solo puede ser evaluado por la persona guardiana del test con allow_test=True."
        )

    npz_path = Path(path) if path is not None else PROCESSED_DIR / "beats.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de latidos en: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if "split" not in data:
            raise KeyError(f"El archivo {npz_path} no contiene la clave 'split'")

        split_col = np.asarray(data["split"])
        # Decodificar bytes si fuese necesario (ej. si fue guardado como bytes)
        if split_col.dtype.kind in {"S", "a"}:
            split_strings = np.char.decode(split_col)
        else:
            split_strings = split_col.astype(str)

        mask = np.isin(split_strings, list(splits))

        filtered_dict: dict[str, np.ndarray] = {
            key: np.asarray(data[key])[mask] for key in data.files
        }

    return filtered_dict


def validate_beats(data: Mapping[str, Any]) -> bool:
    """Verifica que un diccionario de arrays cumpla rigurosamente con el Contrato C1.

    Parámetros
    ----------
    data:
        Mapeo con los arrays correspondientes a C1.

    Retorna
    -------
    bool:
        True si la estructura cumple con todas las especificaciones de C1.

    Excepciones
    -----------
    ValueError | TypeError:
        Si alguna forma, tipo o restricción de valor es violada.
    """
    missing_keys = C1_REQUIRED_KEYS - set(data.keys())
    if missing_keys:
        raise ValueError(f"Faltan claves requeridas por el Contrato C1: {sorted(missing_keys)}")

    # Obtener el número de latidos n
    x_arr = np.asarray(data["X"])
    if x_arr.ndim != 2 or x_arr.shape[1] != 256:
        raise ValueError(f"'X' debe tener forma (n, 256), pero tiene {x_arr.shape}")
    if x_arr.dtype != np.float32:
        raise TypeError(f"'X' debe ser float32, se encontró {x_arr.dtype}")

    n = x_arr.shape[0]

    # Validar longitudes alineadas
    for key in C1_REQUIRED_KEYS:
        arr = np.asarray(data[key])
        if arr.shape[0] != n:
            raise ValueError(
                f"Array '{key}' no alineado: longitud {arr.shape[0]} diferente de n={n}"
            )

    # Validar symbol
    symbol_arr = np.asarray(data["symbol"])
    if symbol_arr.ndim != 1:
        raise ValueError(f"'symbol' debe tener 1 dimensión, forma actual: {symbol_arr.shape}")
    if symbol_arr.dtype.kind not in {"U", "S"}:
        raise TypeError(f"'symbol' debe ser de tipo string (<U1), encontrado: {symbol_arr.dtype}")

    # Validar aami
    aami_arr = np.asarray(data["aami"])
    if aami_arr.ndim != 1:
        raise ValueError(f"'aami' debe tener 1 dimensión, forma actual: {aami_arr.shape}")
    if aami_arr.dtype.kind not in {"U", "S"}:
        raise TypeError(f"'aami' debe ser de tipo string (<U1), encontrado: {aami_arr.dtype}")
    unique_aami = set(np.unique(aami_arr.astype(str)))
    invalid_aami = unique_aami - ALLOWED_AAMI
    if invalid_aami:
        raise ValueError(f"Valores AAMI inválidos encontrados: {invalid_aami}")

    # Validar label
    label_arr = np.asarray(data["label"])
    if label_arr.ndim != 1:
        raise ValueError(f"'label' debe tener 1 dimensión, forma actual: {label_arr.shape}")
    if label_arr.dtype != np.int8:
        raise TypeError(f"'label' debe ser int8, se encontró {label_arr.dtype}")
    unique_labels = set(np.unique(label_arr))
    invalid_labels = unique_labels - {0, 1}
    if invalid_labels:
        raise ValueError(f"Valores en 'label' deben ser solo 0 o 1, se encontró {invalid_labels}")

    # Consistencia label == 0 <==> symbol == 'N'
    sym_str = symbol_arr.astype(str)
    expected_labels = np.where(sym_str == "N", np.int8(0), np.int8(1))
    if not np.array_equal(label_arr, expected_labels):
        mismatches = np.sum(label_arr != expected_labels)
        raise ValueError(
            f"Inconsistencia en 'label': {mismatches} latidos no cumplen label=0 <=> symbol=='N'"
        )

    # Validar record
    record_arr = np.asarray(data["record"])
    if record_arr.ndim != 1:
        raise ValueError(f"'record' debe tener 1 dimensión, forma actual: {record_arr.shape}")
    if record_arr.dtype != np.int16:
        raise TypeError(f"'record' debe ser int16, se encontró {record_arr.dtype}")

    # Validar split
    split_arr = np.asarray(data["split"])
    if split_arr.ndim != 1:
        raise ValueError(f"'split' debe tener 1 dimensión, forma actual: {split_arr.shape}")
    if split_arr.dtype.kind not in {"U", "S"}:
        raise TypeError(f"'split' debe ser string (<U5), encontrado: {split_arr.dtype}")
    unique_splits = set(np.unique(split_arr.astype(str)))
    invalid_splits = unique_splits - ALLOWED_SPLITS
    if invalid_splits:
        raise ValueError(f"Valores de split inválidos encontrados: {invalid_splits}")

    # Validar r_sample
    r_arr = np.asarray(data["r_sample"])
    if r_arr.ndim != 1:
        raise ValueError(f"'r_sample' debe tener 1 dimensión, forma actual: {r_arr.shape}")
    if r_arr.dtype != np.int32:
        raise TypeError(f"'r_sample' debe ser int32, se encontró {r_arr.dtype}")

    # Validar rr_prev y rr_next
    for rr_key in ("rr_prev", "rr_next"):
        rr_arr = np.asarray(data[rr_key])
        if rr_arr.ndim != 1:
            raise ValueError(f"'{rr_key}' debe tener 1 dimensión, forma: {rr_arr.shape}")
        if rr_arr.dtype != np.float32:
            raise TypeError(f"'{rr_key}' debe ser float32, se encontró {rr_arr.dtype}")

    # Validar mlii_noisy
    noisy_arr = np.asarray(data["mlii_noisy"])
    if noisy_arr.ndim != 1:
        raise ValueError(f"'mlii_noisy' debe tener 1 dimensión, forma: {noisy_arr.shape}")
    if noisy_arr.dtype != np.bool_:
        raise TypeError(f"'mlii_noisy' debe ser bool, se encontró {noisy_arr.dtype}")

    return True
