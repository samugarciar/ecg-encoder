"""Umbral de anomalía y métricas de detección a partir de C2 (contrato C3).

El umbral es el percentil 95 del error de los latidos N de val (D-010). Se ajusta siempre sobre
val y se aplica tal cual al split evaluado: un latido es anómalo si su error supera el umbral.
La clase positiva es "anómalo" (label = 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg.config import AAMI_NORMAL, NORMAL_SYMBOLS
from ecg.score import validate_scores

DEFAULT_PERCENTILE: float = 95.0  # D-010
AAMI_ANOMALOUS: tuple[str, ...] = ("S", "V", "F", "Q")


def check_scores(scores: Mapping[str, np.ndarray]) -> None:
    """Valida C2 y que `label` sea 0 exactamente cuando `symbol == "N"` (D-005).

    Si no se cumple, se detiene: nunca se recalcula la etiqueta para que cuadre (regla 11).
    """
    validate_scores(scores)
    symbol = np.asarray(scores["symbol"]).astype(str)
    expected = np.where(np.isin(symbol, list(NORMAL_SYMBOLS)), 0, 1)
    mismatches = int(np.sum(np.asarray(scores["label"]) != expected))
    if mismatches:
        raise ValueError(f"{mismatches} latidos de C2 no cumplen label = 0 <=> symbol == 'N'")


def fit_threshold(val_scores: Mapping[str, np.ndarray], percentile: float) -> float:
    """Percentil del error de los latidos normales (label = 0) de val."""
    normal_error = np.asarray(val_scores["error"])[np.asarray(val_scores["label"]) == 0]
    if len(normal_error) == 0:
        raise ValueError("No hay latidos normales en val para ajustar el umbral")
    return float(np.percentile(normal_error, percentile))


def _ratio(num: int, den: int) -> float | None:
    """num / den, o None si el denominador es 0 (la métrica no está definida)."""
    return num / den if den else None


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Matriz de confusión con "anómalo" como clase positiva."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
    }


def detection_metrics(
    y_true: np.ndarray, error: np.ndarray, threshold: float
) -> dict[str, float | None]:
    """AUC-ROC, AUC-PR y las métricas del umbral sobre un conjunto de latidos.

    Las AUC usan el error continuo y no dependen del umbral. AUC-PR es la precisión promedio
    (`average_precision_score`). Una métrica sin definir (por ejemplo, una AUC con una sola
    clase) queda en None.
    """
    y_true = np.asarray(y_true).astype(int)
    error = np.asarray(error)
    c = confusion(y_true, error > threshold)
    both_classes = len(np.unique(y_true)) == 2
    precision = _ratio(c["tp"], c["tp"] + c["fp"])
    sensitivity = _ratio(c["tp"], c["tp"] + c["fn"])
    if precision is None or sensitivity is None:
        f1 = None
    else:
        f1 = _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])
    return {
        "auc_roc": float(roc_auc_score(y_true, error)) if both_classes else None,
        "auc_pr": float(average_precision_score(y_true, error)) if both_classes else None,
        "sensitivity": sensitivity,
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "precision": precision,
        "f1": f1,
        "accuracy": _ratio(c["tp"] + c["tn"], len(y_true)),
    }


def evaluate_scores(scores: Mapping[str, np.ndarray], threshold: float) -> dict[str, Any]:
    """Bloques de métricas de C3 (n, confusion, global, sensibilidades y vistas) para un split."""
    check_scores(scores)
    error = np.asarray(scores["error"])
    label = np.asarray(scores["label"]).astype(int)
    symbol = np.asarray(scores["symbol"]).astype(str)
    aami = np.asarray(scores["aami"]).astype(str)
    clean = ~np.asarray(scores["mlii_noisy"])
    pred = error > threshold

    per_symbol = {
        sym: float(pred[symbol == sym].mean())
        for sym in sorted(set(symbol) - NORMAL_SYMBOLS)
    }
    per_aami = {
        cls: float(pred[aami == cls].mean()) if np.any(aami == cls) else None
        for cls in AAMI_ANOMALOUS
    }
    # D-005: la vista AAMI cuenta N, L, R, e y j como normales; el umbral y las predicciones
    # son los mismos, solo cambia la verdad terreno
    aami_label = (~np.isin(symbol, list(AAMI_NORMAL))).astype(int)

    return {
        "n": {"normal": int(np.sum(label == 0)), "anomalous": int(np.sum(label == 1))},
        "confusion": confusion(label, pred),
        "global": detection_metrics(label, error, threshold),
        "per_symbol_sensitivity": per_symbol,
        "per_aami_sensitivity": per_aami,
        "aami_view": detection_metrics(aami_label, error, threshold),
        "clean_only": detection_metrics(label[clean], error[clean], threshold),
    }


def build_results(
    run_id: str,
    split: str,
    data_version: str | None,
    val_scores: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    percentile: float = DEFAULT_PERCENTILE,
) -> dict[str, Any]:
    """Resultado completo del contrato C3: umbral ajustado en val y métricas sobre `scores`."""
    check_scores(val_scores)
    threshold = fit_threshold(val_scores, percentile)
    return {
        "run_id": run_id,
        "split": split,
        "data_version": data_version,
        "threshold": {
            "method": "percentile",
            "percentile": percentile,
            "value": threshold,
            "fitted_on": "val",
        },
        **evaluate_scores(scores, threshold),
    }
