"""Pruebas del entrenamiento y del cálculo de C2 sobre un C1 sintético pequeño."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ecg.models.preprocessing import normal_mask
from ecg.score import save_scores, score_split, validate_scores
from ecg.train import make_run_id, train_autoencoder
from scripts.make_mock_beats import save_mock_beats


def _config(name: str) -> dict:
    """Configuración mínima para que el entrenamiento tarde segundos."""
    return {
        "name": name,
        "seed": 0,
        "device": "cpu",
        "data": {"exclude_noisy": True, "normalization": "zscore"},
        "model": {"input_length": 256, "channels": [4, 8], "kernel_size": 5, "latent_dim": 4},
        "training": {"batch_size": 64, "learning_rate": 0.001, "max_epochs": 3, "patience": 1,
                     "min_delta": 0.0},
    }


@pytest.fixture()
def beats_path(tmp_path: Path) -> Path:
    npz_path, _ = save_mock_beats(target_dir=tmp_path, filename_prefix="beats", n_beats=300, seed=1)
    return npz_path


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def test_run_id_follows_convention() -> None:
    run_id = make_run_id("ana", "baseline", datetime(2026, 10, 22, 9, 5))
    assert run_id == "20261022-0905_ana_baseline"


def test_train_creates_run_with_clean_normals(tmp_path: Path, beats_path: Path) -> None:
    run_dir = train_autoencoder(_config("a"), "test", beats_path, tmp_path / "runs", verbose=False)
    for name in ("config.yaml", "model.pt", "history.json", "meta.json"):
        assert (run_dir / name).exists(), name

    beats = _load(beats_path)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_train"] == int(normal_mask(beats, "train", exclude_noisy=True).sum())
    assert meta["n_val"] == int(normal_mask(beats, "val", exclude_noisy=True).sum())
    assert 1 <= meta["epochs_run"] <= 3
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert meta["best_val_loss"] == min(h["val_loss"] for h in history)


def test_training_is_deterministic(tmp_path: Path, beats_path: Path) -> None:
    """Misma semilla y mismos datos: las mismas pérdidas época por época (regla 9)."""
    runs = tmp_path / "runs"
    first = train_autoencoder(_config("a"), "test", beats_path, runs, verbose=False)
    second = train_autoencoder(_config("b"), "test", beats_path, runs, verbose=False)
    loss_a = json.loads((first / "history.json").read_text(encoding="utf-8"))
    loss_b = json.loads((second / "history.json").read_text(encoding="utf-8"))
    assert [h["val_loss"] for h in loss_a] == [h["val_loss"] for h in loss_b]


def test_score_split_follows_c2(tmp_path: Path, beats_path: Path) -> None:
    run_dir = train_autoencoder(_config("a"), "test", beats_path, tmp_path / "runs", verbose=False)
    scores = score_split(run_dir, "val", beats_path)
    assert validate_scores(scores)

    beats = _load(beats_path)
    assert np.all(beats["split"][scores["idx"]] == "val")
    assert len(scores["idx"]) == int((beats["split"] == "val").sum())  # incluye los ruidosos
    assert np.array_equal(scores["symbol"], beats["symbol"][scores["idx"]])
    assert save_scores(run_dir, "val", scores).name == "scores_val.npz"


def test_score_split_blocks_test(tmp_path: Path, beats_path: Path) -> None:
    run_dir = train_autoencoder(_config("a"), "test", beats_path, tmp_path / "runs", verbose=False)
    with pytest.raises(PermissionError):
        score_split(run_dir, "test", beats_path)
