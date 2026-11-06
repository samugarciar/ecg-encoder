"""Exportación de un run a la interfaz web (scripts/export_web_model.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from ecg.evaluate.metrics import fit_threshold
from ecg.models.autoencoder import ConvAutoencoder
from ecg.paths import REPO_ROOT
from ecg.score import save_scores, score_split
from ecg.train import train_autoencoder
from ecg.utils.seed import seed_everything
from scripts import validate_r_detector
from scripts.export_web_model import export_onnx, export_run, preprocessing_spec
from scripts.make_mock_beats import save_mock_beats
from scripts.validate_r_detector import match_beats

WEB_MODEL = REPO_ROOT / "web" / "model"


def _onnx_forward(path: Path, x: np.ndarray) -> np.ndarray:
    return ReferenceEvaluator(onnx.load(str(path))).run(None, {"beats": x})[0]


def _train_tiny(tmp_path: Path, beats_path: Path) -> Path:
    """Run mínimo sobre el C1 sintético, para que la prueba tarde segundos."""
    config = {
        "name": "web", "seed": 0, "device": "cpu",
        "data": {"exclude_noisy": True, "normalization": "zscore"},
        "model": {"input_length": 256, "channels": [4, 8], "kernel_size": 5, "latent_dim": 4},
        "training": {"batch_size": 64, "learning_rate": 0.001, "max_epochs": 2, "patience": 1,
                     "min_delta": 0.0},
    }
    return train_autoencoder(config, "test", beats_path, tmp_path / "runs", verbose=False)


@pytest.mark.parametrize("batch", [1, 7])
def test_onnx_matches_torch(tmp_path: Path, batch: int) -> None:
    """El ONNX reconstruye igual que PyTorch, con cualquier tamaño de lote."""
    seed_everything(0)
    model = ConvAutoencoder(channels=(4, 8), kernel_size=5, latent_dim=4).eval()
    x = np.random.default_rng(0).normal(size=(batch, 1, 256)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    path = export_onnx(model, tmp_path / "model.onnx")
    onnx.checker.check_model(onnx.load(str(path)))
    assert np.allclose(_onnx_forward(path, x), expected, atol=1e-5)


def test_export_run_uses_val_threshold(tmp_path: Path) -> None:
    beats_path, _ = save_mock_beats(tmp_path, "beats", n_beats=300, seed=1)
    run_dir = _train_tiny(tmp_path, beats_path)
    val = score_split(run_dir, "val", beats_path)
    save_scores(run_dir, "val", val)

    out = export_run(run_dir, tmp_path / "web", results_dir=tmp_path / "results")
    card = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert card["run_id"] == run_dir.name
    assert card["is_demo"] is False
    assert card["threshold"]["value"] == pytest.approx(fit_threshold(val, 95.0))
    assert card["threshold"]["fitted_on"] == "val"
    assert card["normalization"] == "zscore"
    assert (out / "model.onnx").exists() and (out / "preprocessing.json").exists()


def test_export_run_without_scores_explains_what_to_run(tmp_path: Path) -> None:
    beats_path, _ = save_mock_beats(tmp_path, "beats", n_beats=300, seed=1)
    run_dir = _train_tiny(tmp_path, beats_path)
    with pytest.raises(FileNotFoundError, match="ecg.score"):
        export_run(run_dir, tmp_path / "web", results_dir=tmp_path / "results")


def test_match_beats_counts_hits_misses_and_extras() -> None:
    reference = np.array([100, 400, 700, 1000])
    detected = np.array([103, 395, 560, 1000, 1300])  # 700 sin detectar; 560 y 1300 sobran
    m = match_beats(reference, detected, tolerance=54)
    assert (m["tp"], m["fn"], m["fp"]) == (3, 1, 2)
    assert m["se"] == pytest.approx(0.75) and m["ppv"] == pytest.approx(0.6)
    assert m["median_offset"] == 0


def test_validate_r_detector_refuses_test_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["validate_r_detector", "--records", "101", "100"])
    with pytest.raises(PermissionError, match="100"):
        validate_r_detector.main()


def test_preprocessing_json_is_in_sync_with_config() -> None:
    """web/model/preprocessing.json sale de config.py (D-015); si cambia config, se regenera."""
    committed = json.loads((WEB_MODEL / "preprocessing.json").read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(preprocessing_spec()))
    # Los coeficientes del filtro cambian en el último decimal según la plataforma (Windows o
    # Linux): se comparan con tolerancia; el resto debe ser idéntico
    for key in ("sos", "zi"):
        assert np.allclose(committed["filter"].pop(key), expected["filter"].pop(key),
                           rtol=1e-12, atol=1e-14), key
    assert committed == expected


def test_published_model_is_consistent() -> None:
    """El modelo que se despliega en web/model/ carga, tiene la forma de C1 y un umbral válido."""
    card = json.loads((WEB_MODEL / "model.json").read_text(encoding="utf-8"))
    assert card["input_length"] == 256
    assert card["threshold"]["fitted_on"] == "val"
    assert card["threshold"]["value"] > 0
    assert isinstance(card["is_demo"], bool)
    x = np.zeros((2, 1, 256), dtype=np.float32)
    assert _onnx_forward(WEB_MODEL / card["onnx"]["file"], x).shape == (2, 1, 256)
