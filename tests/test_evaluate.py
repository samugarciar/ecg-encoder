"""Pruebas del umbral y de las métricas de evaluación (contrato C3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg.config import AAMI_MAP
from ecg.evaluate.__main__ import evaluate_run
from ecg.evaluate.metrics import build_results, evaluate_scores, fit_threshold
from scripts.make_mock_beats import generate_mock_beats
from scripts.make_mock_scores import generate_mock_scores

C3_KEYS = {
    "run_id", "split", "data_version", "threshold", "n", "confusion", "global",
    "per_symbol_sensitivity", "per_aami_sensitivity", "aami_view", "clean_only",
}
METRIC_KEYS = {"auc_roc", "auc_pr", "sensitivity", "specificity", "precision", "f1", "accuracy"}


def _scores(error: list[float], symbol: list[str], noisy: list[bool] | None = None) -> dict:
    """C2 mínimo a partir del error y el símbolo de cada latido."""
    m = len(error)
    return {
        "idx": np.arange(m, dtype=np.int64),
        "error": np.array(error, dtype=np.float32),
        "label": np.array([0 if s == "N" else 1 for s in symbol], dtype=np.int8),
        "symbol": np.array(symbol, dtype="<U1"),
        "aami": np.array([AAMI_MAP[s] for s in symbol], dtype="<U1"),
        "record": np.full(m, 116, dtype=np.int16),
        "mlii_noisy": np.array(noisy if noisy is not None else [False] * m, dtype=np.bool_),
    }


# Caso calculado a mano con umbral 0.5: se marcan 0.6 (N), 0.9 (V), 0.7 (L) y 0.8 (A)
HAND_ERROR = [0.1, 0.2, 0.6, 0.3, 0.9, 0.4, 0.7, 0.8]
HAND_SYMBOL = ["N", "N", "N", "N", "V", "V", "L", "A"]


def test_fit_threshold_uses_only_val_normals() -> None:
    error = list(range(1, 101)) + [1000.0] * 10
    scores = _scores(error, ["N"] * 100 + ["V"] * 10)
    assert fit_threshold(scores, 95.0) == pytest.approx(95.05)


def test_global_metrics_match_hand_count() -> None:
    result = evaluate_scores(_scores(HAND_ERROR, HAND_SYMBOL), threshold=0.5)
    assert result["n"] == {"normal": 4, "anomalous": 4}
    assert result["confusion"] == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}
    g = result["global"]
    assert g["sensitivity"] == pytest.approx(0.75)
    assert g["specificity"] == pytest.approx(0.75)
    assert g["precision"] == pytest.approx(0.75)
    assert g["f1"] == pytest.approx(0.75)
    assert g["accuracy"] == pytest.approx(6 / 8)
    # 15 de los 16 pares (anómalo, normal) tienen el anómalo con más error
    assert g["auc_roc"] == pytest.approx(15 / 16)


def test_per_class_sensitivity() -> None:
    result = evaluate_scores(_scores(HAND_ERROR, HAND_SYMBOL), threshold=0.5)
    assert result["per_symbol_sensitivity"] == {"A": 1.0, "L": 1.0, "V": 0.5}
    # L es N en AAMI: no entra en las sensibilidades AAMI; F y Q no aparecen
    assert result["per_aami_sensitivity"] == {"S": 1.0, "V": 0.5, "F": None, "Q": None}


def test_aami_view_counts_bundle_branch_as_normal() -> None:
    v = evaluate_scores(_scores(HAND_ERROR, HAND_SYMBOL), threshold=0.5)["aami_view"]
    # Con L normal: tp = V(0.9) y A; fn = V(0.4); fp = N(0.6) y L; tn = tres N
    assert v["sensitivity"] == pytest.approx(2 / 3)
    assert v["specificity"] == pytest.approx(3 / 5)
    assert v["precision"] == pytest.approx(0.5)


def test_clean_only_drops_noisy_beats() -> None:
    noisy = [False, False, True, False, False, False, False, False]  # el N con error 0.6
    result = evaluate_scores(_scores(HAND_ERROR, HAND_SYMBOL, noisy), threshold=0.5)
    assert result["global"]["specificity"] == pytest.approx(0.75)
    assert result["clean_only"]["specificity"] == pytest.approx(1.0)
    assert result["clean_only"]["precision"] == pytest.approx(1.0)


def test_undefined_metrics_are_none() -> None:
    g = evaluate_scores(_scores([0.1, 0.9], ["N", "N"]), threshold=0.5)["global"]
    assert g["auc_roc"] is None
    assert g["auc_pr"] is None
    assert g["sensitivity"] is None
    assert g["f1"] is None
    assert g["specificity"] == pytest.approx(0.5)


def test_label_symbol_mismatch_stops() -> None:
    scores = _scores(HAND_ERROR, HAND_SYMBOL)
    scores["label"][0] = 1
    with pytest.raises(ValueError, match="label = 0"):
        evaluate_scores(scores, threshold=0.5)


def test_threshold_is_fitted_on_val_for_any_split() -> None:
    val = _scores([0.1, 0.2, 0.3, 0.4, 5.0], ["N", "N", "N", "N", "V"])
    train = _scores([10.0, 20.0], ["N", "V"])
    results = build_results("r", "train", None, val, train, percentile=50.0)
    assert results["threshold"]["value"] == pytest.approx(0.25)
    assert results["threshold"]["fitted_on"] == "val"
    assert results["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}


@pytest.fixture()
def mock_run(tmp_path: Path) -> Path:
    beats, _ = generate_mock_beats(n_beats=1500, seed=7)
    run_dir = tmp_path / "runs" / "20261104-2000_test_mock"
    run_dir.mkdir(parents=True)
    for split in ("train", "val"):
        np.savez_compressed(
            run_dir / f"scores_{split}.npz", **generate_mock_scores(beats, split=split, seed=7)
        )
    return run_dir


def test_evaluate_run_writes_c3(tmp_path: Path, mock_run: Path) -> None:
    path, results = evaluate_run(mock_run, "val", results_dir=tmp_path / "results")
    assert path.name == "20261104-2000_test_mock__val.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == results
    assert set(saved) == C3_KEYS
    for block in ("global", "aami_view", "clean_only"):
        assert set(saved[block]) == METRIC_KEYS
    assert set(saved["confusion"]) == {"tp", "fp", "fn", "tn"}
    assert set(saved["per_aami_sensitivity"]) == {"S", "V", "F", "Q"}
    assert saved["data_version"] is None  # el run mock no tiene meta.json
    n = saved["n"]["normal"] + saved["n"]["anomalous"]
    assert sum(saved["confusion"].values()) == n
    # Con errores lognormales tan separados, el detector debe ser casi perfecto
    assert saved["global"]["auc_roc"] > 0.95


def test_evaluate_run_train_reuses_val_threshold(tmp_path: Path, mock_run: Path) -> None:
    _, val = evaluate_run(mock_run, "val", results_dir=tmp_path)
    _, train = evaluate_run(mock_run, "train", results_dir=tmp_path)
    assert train["threshold"] == val["threshold"]


def test_evaluate_run_blocks_test(tmp_path: Path, mock_run: Path) -> None:
    with pytest.raises(PermissionError):
        evaluate_run(mock_run, "test", results_dir=tmp_path)
