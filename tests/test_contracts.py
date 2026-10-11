"""Pruebas unitarias para validar el cumplimiento estricto de los contratos C1 y C2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg.io import validate_beats
from scripts.make_mock_beats import generate_mock_beats, save_mock_beats
from scripts.make_mock_scores import generate_mock_scores


def test_contract_c1_mock_generation(tmp_path: Path) -> None:
    """Valida que los datos generados por make_mock_beats cumplan el Contrato C1."""
    npz_path, meta_path = save_mock_beats(
        target_dir=tmp_path,
        filename_prefix="beats_test",
        n_beats=200,
        seed=123,
    )

    assert npz_path.exists()
    assert meta_path.exists()

    with np.load(npz_path, allow_pickle=False) as data:
        beats_dict = {k: np.asarray(data[k]) for k in data.files}

    # Validar a través de la función canónica
    assert validate_beats(beats_dict) is True

    # Validar metadatos en JSON
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["fs"] == 360
    assert meta["window"]["total"] == 256
    assert "counts" in meta
    assert set(meta["counts"].keys()) == {"train", "val", "test"}


def test_contract_c1_validation_catches_invalid_data() -> None:
    """Verifica que validate_beats detecte violaciones de forma, tipo y consistencia."""
    valid_data, _ = generate_mock_beats(n_beats=50, seed=42)

    # 1. Error de dimensión en X
    bad_x = valid_data.copy()
    bad_x["X"] = bad_x["X"][:, :100]
    with pytest.raises(ValueError, match="'X' debe tener forma"):
        validate_beats(bad_x)

    # 2. Inconsistencia entre label y symbol (ej. label=1 cuando symbol='N')
    bad_label = valid_data.copy()
    corrupted_labels = bad_label["label"].copy()
    first_n_idx = np.where(bad_label["symbol"] == "N")[0][0]
    corrupted_labels[first_n_idx] = 1
    bad_label["label"] = corrupted_labels
    with pytest.raises(ValueError, match="Inconsistencia en 'label'"):
        validate_beats(bad_label)

    # 3. Longitud no alineada
    bad_len = valid_data.copy()
    bad_len["record"] = bad_len["record"][:-1]
    with pytest.raises(ValueError, match="no alineado"):
        validate_beats(bad_len)


def test_contract_c2_mock_scores(tmp_path: Path) -> None:
    """Valida que las puntuaciones generadas por make_mock_scores cumplan el Contrato C2."""
    beats, _ = generate_mock_beats(n_beats=300, seed=99)
    scores = generate_mock_scores(beats, split="val", seed=99)

    required_keys = {"idx", "error", "label", "symbol", "aami", "record", "mlii_noisy"}
    assert required_keys.issubset(scores.keys())

    m = len(scores["idx"])
    assert m > 0

    # Validar formas y tipos exactos
    assert scores["idx"].shape == (m,)
    assert scores["idx"].dtype == np.int64

    assert scores["error"].shape == (m,)
    assert scores["error"].dtype == np.float32
    assert np.all(scores["error"] >= 0.0)

    assert scores["label"].shape == (m,)
    assert scores["label"].dtype == np.int8

    assert scores["symbol"].shape == (m,)
    assert scores["symbol"].dtype.kind in {"U", "S"}

    assert scores["aami"].shape == (m,)
    assert scores["aami"].dtype.kind in {"U", "S"}

    assert scores["record"].shape == (m,)
    assert scores["record"].dtype == np.int16

    assert scores["mlii_noisy"].shape == (m,)
    assert scores["mlii_noisy"].dtype == np.bool_

    # Verificar que el error promedio de anómalos sea mayor al de normales
    norm_errors = scores["error"][scores["label"] == 0]
    anom_errors = scores["error"][scores["label"] == 1]
    if len(anom_errors) > 0 and len(norm_errors) > 0:
        assert np.mean(anom_errors) > np.mean(norm_errors)
