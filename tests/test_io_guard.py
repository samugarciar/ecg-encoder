"""Pruebas unitarias para la barrera de seguridad de acceso al split de test (DS2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg.io import load_beats


@pytest.fixture
def dummy_beats_file(tmp_path: Path) -> Path:
    """Crea un archivo .npz mínimo en un directorio temporal para pruebas de I/O."""
    beats_file = tmp_path / "beats.npz"
    splits = np.array(["train", "val", "test", "train", "val"], dtype="<U5")
    x = np.zeros((5, 256), dtype=np.float32)
    labels = np.array([0, 0, 1, 0, 0], dtype=np.int8)

    np.savez_compressed(beats_file, split=splits, X=x, label=labels)
    return beats_file


def test_load_beats_test_split_raises_permission_error(dummy_beats_file: Path) -> None:
    """Verifica que solicitar el split 'test' sin allow_test=True lance PermissionError."""
    with pytest.raises(PermissionError, match="El split 'test' .* está prohibido"):
        load_beats(path=dummy_beats_file, splits=("train", "test"), allow_test=False)

    with pytest.raises(PermissionError, match="El split 'test' .* está prohibido"):
        load_beats(path=dummy_beats_file, splits=["test"], allow_test=False)


def test_load_beats_test_split_allowed_with_explicit_flag(dummy_beats_file: Path) -> None:
    """Verifica que solicitar el split 'test' con allow_test=True sea permitido."""
    data = load_beats(path=dummy_beats_file, splits=("test",), allow_test=True)
    assert "split" in data
    assert len(data["split"]) == 1
    assert data["split"][0] == "test"


def test_load_beats_default_splits_succeed(dummy_beats_file: Path) -> None:
    """Verifica que la carga por defecto ('train', 'val') funcione correctamente sin 'test'."""
    data = load_beats(path=dummy_beats_file, splits=("train", "val"))
    assert len(data["split"]) == 4
    assert set(data["split"]) == {"train", "val"}
    assert "test" not in data["split"]


def test_load_beats_nonexistent_file_raises_error(tmp_path: Path) -> None:
    """Verifica que intentar cargar un archivo inexistente lance FileNotFoundError."""
    nonexistent = tmp_path / "no_existe.npz"
    with pytest.raises(FileNotFoundError):
        load_beats(path=nonexistent, splits=("train", "val"))
