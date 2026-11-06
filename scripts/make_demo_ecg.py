"""Genera un ECG continuo SINTÉTICO de ejemplo para la interfaz web (web/samples/).

Uso: python -m scripts.make_demo_ecg
Escribe ecg_sintetico.csv con el mismo formato que los CSV de MIT-BIH (columnas 'sample #',
'MLII', 'V5' en unidades del conversor A/D) y ecg_sintetico_annotations.txt con el formato de
sus anotaciones. Los latidos usan las formas de scripts/make_mock_beats.py, con latidos
prematuros V y A, deriva de línea base y ruido de red, para que el filtro tenga algo que hacer.
No proviene de ningún paciente: se usa en lugar de registros reales para no publicar datos de
MIT-BIH y para no tocar nunca los registros de test (regla 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecg.config import ADC_GAIN, ADC_ZERO, FS, WIN_LEN, WIN_PRE
from ecg.paths import REPO_ROOT
from ecg.utils.seed import seed_everything
from scripts.make_mock_beats import _synthetic_beat

SAMPLES_DIR: Path = REPO_ROOT / "web" / "samples"
NAME: str = "ecg_sintetico"
ANN_HEADER: str = "      Time   Sample #  Type  Sub Chan  Num\tAux"


def _rhythm(n_beats: int, rng: np.random.Generator) -> list[str]:
    """Símbolos de los latidos: N con algunos V (con pausa compensatoria) y A aislados."""
    symbols = ["N"] * n_beats
    for i in rng.choice(np.arange(4, n_beats - 2), size=n_beats // 9, replace=False):
        if symbols[i - 1] == "N" and symbols[i + 1] == "N":
            symbols[i] = "V" if rng.random() < 0.6 else "A"
    return symbols


def synthetic_ecg(
    seconds: float = 60.0, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Devuelve (MLII en mV, muestras de los R, símbolos) de un ECG sintético a FS Hz."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    base_rr = 0.83 * FS
    symbols = _rhythm(int(seconds / 0.83) + 4, rng)

    r_samples: list[int] = []
    kept: list[str] = []
    r = int(0.6 * FS)
    for i, sym in enumerate(symbols):
        if r + WIN_LEN >= n:
            break
        r_samples.append(r)
        kept.append(sym)
        nxt = symbols[i + 1] if i + 1 < len(symbols) else "N"
        if nxt == "V":
            step = 0.62 * base_rr  # prematuro
        elif nxt == "A":
            step = 0.72 * base_rr
        elif sym == "V":
            step = 1.38 * base_rr  # pausa compensatoria
        else:
            step = base_rr
        r += int(step + rng.normal(0, 0.02 * FS))

    t_axis = np.arange(WIN_LEN, dtype=np.float32)
    x = np.zeros(n)
    for r, sym in zip(r_samples, kept, strict=True):
        x[r - WIN_PRE:r - WIN_PRE + WIN_LEN] += _synthetic_beat(t_axis, sym, rng)

    t = np.arange(n) / FS
    x += 0.25 * np.sin(2 * np.pi * 0.22 * t) + 0.08 * np.sin(2 * np.pi * 0.05 * t + 1.0)
    x += 0.03 * np.sin(2 * np.pi * 60.0 * t)  # red eléctrica
    x += rng.normal(0, 0.01, n)
    return x, np.asarray(r_samples), kept


def _timestamp(sample: int) -> str:
    seconds = sample / FS
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def save_demo_ecg(out_dir: Path | str = SAMPLES_DIR, seconds: float = 60.0, seed: int = 7
                  ) -> tuple[Path, Path]:
    """Escribe el CSV y las anotaciones en `out_dir`; devuelve sus rutas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlii, r_samples, symbols = synthetic_ecg(seconds, seed)
    adc_mlii = np.round(mlii * ADC_GAIN + ADC_ZERO).astype(int)
    adc_v5 = np.round(0.45 * mlii * ADC_GAIN + ADC_ZERO).astype(int)

    csv_path = out_dir / f"{NAME}.csv"
    lines = ["'sample #','MLII','V5'"]
    lines += [f"{i},{a},{b}" for i, (a, b) in enumerate(zip(adc_mlii, adc_v5, strict=True))]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ann_path = out_dir / f"{NAME}_annotations.txt"
    rows = [ANN_HEADER, f"{_timestamp(18):>12}{18:>9}{'+':>6}{0:>5}{0:>5}{0:>5}\t(N"]
    rows += [f"{_timestamp(r):>12}{r:>9}{s:>6}{0:>5}{0:>5}{0:>5}"
             for r, s in zip(r_samples, symbols, strict=True)]
    ann_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path, ann_path


def main() -> None:
    csv_path, ann_path = save_demo_ecg()
    print(f"ECG sintético escrito en:\n  - {csv_path}\n  - {ann_path}")


if __name__ == "__main__":
    main()
