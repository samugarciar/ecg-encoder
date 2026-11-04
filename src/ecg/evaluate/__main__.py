"""Evalúa un run a partir de sus puntajes C2 y escribe el contrato C3.

Uso: python -m ecg.evaluate --run runs/<run_id> --split val
Lee runs/<run_id>/scores_val.npz (umbral) y scores_<split>.npz (métricas), y escribe
results/<run_id>__<split>.json. Desde aquí solo se evalúan train y val: el split de test lo
evalúa src/ecg/evaluate/final_test.py (regla 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg.evaluate.metrics import DEFAULT_PERCENTILE, build_results
from ecg.paths import RESULTS_DIR

DEV_SPLITS: tuple[str, ...] = ("train", "val")


def load_scores(run_dir: Path | str, split: str) -> dict[str, np.ndarray]:
    """Carga runs/<run_id>/scores_<split>.npz."""
    path = Path(run_dir) / f"scores_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}: córrelo antes con python -m ecg.score")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def run_data_version(run_dir: Path | str) -> str | None:
    """data_version de C1 con que se entrenó el run (meta.json del run), si existe."""
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8")).get("data_version")


def evaluate_run(
    run_dir: Path | str,
    split: str = "val",
    percentile: float = DEFAULT_PERCENTILE,
    results_dir: Path | str = RESULTS_DIR,
) -> tuple[Path, dict[str, Any]]:
    """Evalúa un split de desarrollo de un run y guarda results/<run_id>__<split>.json."""
    if split not in DEV_SPLITS:
        raise PermissionError(
            f"Split '{split}' no permitido: aquí solo se evalúan {DEV_SPLITS}. "
            "El test se evalúa con src/ecg/evaluate/final_test.py (regla 4)."
        )
    run_dir = Path(run_dir)
    results = build_results(
        run_id=run_dir.name,
        split=split,
        data_version=run_data_version(run_dir),
        val_scores=load_scores(run_dir, "val"),
        scores=load_scores(run_dir, split),
        percentile=percentile,
    )
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run_dir.name}__{split}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, results


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula el contrato C3 de un run.")
    parser.add_argument("--run", required=True, help="carpeta runs/<run_id>")
    parser.add_argument("--split", default="val", choices=DEV_SPLITS)
    parser.add_argument(
        "--percentile", type=float, default=DEFAULT_PERCENTILE,
        help="percentil del error de los N de val que fija el umbral (D-010)",
    )
    args = parser.parse_args()
    path, results = evaluate_run(args.run, args.split, args.percentile)
    g = results["global"]
    print(
        f"{path}: umbral {results['threshold']['value']:.4f} · AUC-ROC {_fmt(g['auc_roc'])} · "
        f"sensibilidad {_fmt(g['sensitivity'])} · especificidad {_fmt(g['specificity'])} · "
        f"F1 {_fmt(g['f1'])}"
    )


if __name__ == "__main__":
    main()
