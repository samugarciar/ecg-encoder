"""Exporta un run entrenado a la interfaz web (web/model/): ONNX, umbral y preprocesamiento.

Uso, desde la raíz del repositorio:
    python -m scripts.export_web_model --run runs/<run_id>    # modelo real
    python -m scripts.export_web_model --demo                 # modelo de demostración sintético

Escribe en web/model/:
- model.onnx: el autoencoder, con entrada `beats` (lote, 1, 256) y salida `reconstruction`.
- model.json: run_id, umbral (percentil 95 de los N de val, D-010), normalización y las métricas
  de validación (C3) que la interfaz muestra al médico.
- preprocessing.json: fs, conversión A/D, ventana y coeficientes del filtro, sacados de
  config.py (D-015) para que el navegador repita exactamente el pipeline de C1.

Solo usa train y val: el umbral sale de val y nunca se toca el split de test (regla 4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal
from torch import nn

from ecg import config as ecg_config
from ecg.evaluate.metrics import DEFAULT_PERCENTILE, build_results, fit_threshold
from ecg.models.autoencoder import ConvAutoencoder, Upsample2x, upsample2x_linear
from ecg.paths import CONFIGS_DIR, REPO_ROOT, RESULTS_DIR, RUNS_DIR
from ecg.score import load_run, score_split
from ecg.train import load_config, train_autoencoder
from scripts.make_mock_beats import save_mock_beats

WEB_MODEL_DIR: Path = REPO_ROOT / "web" / "model"
DEMO_CONFIG: Path = CONFIGS_DIR / "train" / "demo_web.yaml"
ONNX_OPSET: int = 17
INPUT_NAME, OUTPUT_NAME = "beats", "reconstruction"


class _ShiftUpsample(nn.Module):
    """Upsample2x escrito con sumas: misma cuenta, pero solo con operadores ONNX básicos."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample2x_linear(x)


def _replace_upsample(module: nn.Module) -> None:
    """Cambia cada Upsample2x por _ShiftUpsample, para no depender del operador Resize de ONNX."""
    for name, child in module.named_children():
        if isinstance(child, Upsample2x):
            setattr(module, name, _ShiftUpsample())
        else:
            _replace_upsample(child)


def export_onnx(model: ConvAutoencoder, path: Path | str) -> Path:
    """Exporta el autoencoder a ONNX con el tamaño de lote libre."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    _replace_upsample(model)
    dummy = torch.zeros(2, 1, model.input_length)
    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=ONNX_OPSET, dynamo=False,
    )
    return path


def preprocessing_spec() -> dict[str, Any]:
    """Parámetros del pipeline de C1 que la interfaz necesita, con el filtro ya diseñado.

    El navegador aplica `sosfiltfilt` con estos coeficientes, el estado inicial `zi` y la
    extensión impar de `padlen` muestras, igual que scipy.
    """
    sos = signal.butter(ecg_config.FILTER_ORDER, ecg_config.BANDPASS, btype="bandpass",
                        fs=ecg_config.FS, output="sos")
    ntaps = 2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return {
        "fs": ecg_config.FS,
        "lead": ecg_config.LEAD,
        "adc_zero": ecg_config.ADC_ZERO,
        "adc_gain": ecg_config.ADC_GAIN,
        "window": {"pre": ecg_config.WIN_PRE, "post": ecg_config.WIN_POST,
                   "total": ecg_config.WIN_LEN},
        "filter": {
            "low": ecg_config.BANDPASS[0],
            "high": ecg_config.BANDPASS[1],
            "order": ecg_config.FILTER_ORDER,
            "sos": sos.tolist(),
            "zi": (signal.sosfilt_zi(sos) + 0.0).tolist(),  # + 0.0 quita los -0.0
            "padlen": 3 * ntaps,
        },
        "aami_map": ecg_config.AAMI_MAP,
        "normal_symbols": sorted(ecg_config.NORMAL_SYMBOLS),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return out.stdout.strip()


def _val_results(run_dir: Path, results_dir: Path) -> dict[str, Any]:
    """C3 de val del run: results/<run_id>__val.json, o se calcula con scores_val.npz."""
    path = results_dir / f"{run_dir.name}__val.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    scores_path = run_dir / "scores_val.npz"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"No hay {path} ni {scores_path}. Antes corre:\n"
            f"  python -m ecg.score --run {run_dir} --split val\n"
            f"  python -m ecg.evaluate --run {run_dir} --split val"
        )
    with np.load(scores_path, allow_pickle=False) as data:
        val = {key: np.asarray(data[key]) for key in data.files}
    return build_results(run_dir.name, "val", None, val, val, DEFAULT_PERCENTILE)


def model_card(
    run_dir: Path, run_config: dict[str, Any], results: dict[str, Any], is_demo: bool
) -> dict[str, Any]:
    """Contenido de model.json: lo que la interfaz necesita para decidir y para explicarse."""
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    threshold = results["threshold"]
    if threshold.get("fitted_on") != "val":
        raise ValueError(f"El umbral debe ajustarse en val, no en {threshold.get('fitted_on')}")
    return {
        "run_id": run_dir.name,
        "is_demo": is_demo,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data_version": results.get("data_version") or meta.get("data_version"),
        "input_length": run_config["model"]["input_length"],
        "normalization": run_config["data"]["normalization"],
        "threshold": {
            "value": float(threshold["value"]),
            "percentile": float(threshold["percentile"]),
            "fitted_on": "val",
        },
        "validation": {
            "n": results["n"],
            "global": results["global"],
            "per_aami_sensitivity": results["per_aami_sensitivity"],
        },
        "onnx": {"file": "model.onnx", "input": INPUT_NAME, "output": OUTPUT_NAME,
                 "opset": ONNX_OPSET},
    }


def export_run(
    run_dir: Path | str,
    out_dir: Path | str = WEB_MODEL_DIR,
    results: dict[str, Any] | None = None,
    results_dir: Path | str = RESULTS_DIR,
    is_demo: bool = False,
) -> Path:
    """Escribe model.onnx, model.json y preprocessing.json de un run en `out_dir`."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    model, run_config = load_run(run_dir)
    if run_config["model"]["input_length"] != ecg_config.WIN_LEN:
        raise ValueError(f"El modelo espera {run_config['model']['input_length']} muestras y la "
                         f"ventana de C1 tiene {ecg_config.WIN_LEN}")
    results = results if results is not None else _val_results(run_dir, Path(results_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out_dir / "model.onnx")
    card = model_card(run_dir, run_config, results, is_demo)
    (out_dir / "model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    write_preprocessing(out_dir)
    return out_dir


def write_preprocessing(out_dir: Path | str = WEB_MODEL_DIR) -> Path:
    path = Path(out_dir) / "preprocessing.json"
    path.write_text(json.dumps(preprocessing_spec(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def train_demo(runs_dir: Path | str = RUNS_DIR) -> tuple[Path, dict[str, Any]]:
    """Entrena el modelo de demostración con latidos sintéticos y devuelve (run, C3 de val)."""
    demo_config = load_config(DEMO_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        beats_path, _ = save_mock_beats(Path(tmp), "beats", demo_config["mock"]["n_beats"],
                                        demo_config["seed"])
        run_dir = train_autoencoder(demo_config, "demo", beats_path, runs_dir)
        val = score_split(run_dir, "val", beats_path)
    results = build_results(run_dir.name, "val", "mock", val, val, DEFAULT_PERCENTILE)
    if results["threshold"]["value"] != fit_threshold(val, DEFAULT_PERCENTILE):
        raise ValueError("El umbral del demo no coincide con el percentil de los N de val")
    return run_dir, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta un run a la interfaz web (web/model/).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="carpeta runs/<run_id> con config.yaml y model.pt")
    group.add_argument("--demo", action="store_true",
                       help="entrena y exporta el modelo de demostración con datos sintéticos")
    parser.add_argument("--out", default=str(WEB_MODEL_DIR), help="carpeta de salida")
    args = parser.parse_args()

    if args.demo:
        run_dir, results = train_demo()
        out = export_run(run_dir, args.out, results=results, is_demo=True)
    else:
        out = export_run(args.run, args.out)
    card = json.loads((out / "model.json").read_text(encoding="utf-8"))
    kind = "DEMOSTRACIÓN (sintético)" if card["is_demo"] else "real"
    print(f"{out}: modelo {kind} {card['run_id']} · umbral {card['threshold']['value']:.5f}")


if __name__ == "__main__":
    main()
