"""Valida el detector de picos R de la interfaz (web/js/dsp.js) contra las anotaciones de MIT-BIH.

Uso: python -m scripts.validate_r_detector [--records 101 106 ...]
Requiere Node.js y los datos crudos en data/raw/mitbih_database/. Solo usa registros de DS1
(train y val); los de DS2 (test) se rechazan (regla 4). Para cada registro filtra MLII igual que
C1, corre el detector y empareja cada R detectado con el latido anotado más cercano a <= 150 ms
(tolerancia de la norma ANSI/AAMI EC57). Reporta sensibilidad (Se), valor predictivo positivo
(+P) y el desfase mediano en muestras.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from ecg import config
from ecg.paths import RAW_DIR, REPO_ROOT

TOLERANCE_S: float = 0.150
RUNNER = """
import { readFileSync, writeFileSync } from "node:fs";
import { detectRPeaks } from "./dsp.js";
const [, , inPath, outPath, fs] = process.argv;
const y = Float64Array.from(JSON.parse(readFileSync(inPath, "utf8")));
writeFileSync(outPath, JSON.stringify(Array.from(detectRPeaks(y, Number(fs)))));
"""


def match_beats(reference: np.ndarray, detected: np.ndarray, tolerance: int) -> dict[str, float]:
    """Empareja uno a uno, en orden, latidos anotados y detectados a <= `tolerance` muestras."""
    reference, detected = np.sort(reference), np.sort(detected)
    i = j = tp = 0
    offsets: list[int] = []
    while i < len(reference) and j < len(detected):
        d = int(detected[j] - reference[i])
        if abs(d) <= tolerance:
            tp += 1
            offsets.append(d)
            i += 1
            j += 1
        elif d < 0:
            j += 1  # detección sin latido anotado: falso positivo
        else:
            i += 1  # latido anotado sin detección: falso negativo
    fn, fp = len(reference) - tp, len(detected) - tp
    return {
        "tp": tp, "fn": fn, "fp": fp,
        "se": tp / (tp + fn) if tp + fn else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "median_offset": float(np.median(offsets)) if offsets else float("nan"),
    }


def _load_record(record: int) -> tuple[np.ndarray, np.ndarray]:
    """(MLII filtrada, muestras de los latidos AAMI anotados), como en el notebook de C1."""
    df = pd.read_csv(RAW_DIR / f"{record}.csv")
    leads = [c.strip("'") for c in df.columns[1:]]
    adc = df.iloc[:, 1 + leads.index(config.LEAD)].to_numpy()
    sos = signal.butter(config.FILTER_ORDER, config.BANDPASS, btype="bandpass", fs=config.FS,
                        output="sos")
    y = signal.sosfiltfilt(sos, (adc - config.ADC_ZERO) / config.ADC_GAIN)
    samples = []
    with open(RAW_DIR / f"{record}annotations.txt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fields = line.partition("\t")[0].split()
            if len(fields) >= 3 and fields[2] in config.AAMI_MAP:
                samples.append(int(fields[1]))
    return y, np.asarray(samples)


def detect(y: np.ndarray, node: str, workdir: Path) -> np.ndarray:
    in_path, out_path = workdir / "y.json", workdir / "r.json"
    in_path.write_text(json.dumps(y.tolist()), encoding="utf-8")
    subprocess.run([node, "runner.js", str(in_path), str(out_path), str(config.FS)], cwd=workdir,
                   check=True, capture_output=True, text=True)
    return np.asarray(json.loads(out_path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el detector de picos R de la interfaz.")
    parser.add_argument("--records", type=int, nargs="*",
                        default=sorted(config.DS1_TRAIN + config.DS1_VAL))
    args = parser.parse_args()
    forbidden = set(args.records) - set(config.DS1_TRAIN) - set(config.DS1_VAL)
    if forbidden:
        raise PermissionError(f"Solo se validan registros de DS1; no: {sorted(forbidden)}")
    node = shutil.which("node")
    if node is None:
        raise SystemExit("Se necesita Node.js para correr el detector de web/js/dsp.js.")

    tolerance = int(round(TOLERANCE_S * config.FS))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        shutil.copy(REPO_ROOT / "web" / "js" / "dsp.js", workdir / "dsp.js")
        (workdir / "package.json").write_text('{"type": "module"}', encoding="utf-8")
        (workdir / "runner.js").write_text(RUNNER, encoding="utf-8")
        for record in args.records:
            y, reference = _load_record(record)
            m = match_beats(reference, detect(y, node, workdir), tolerance)
            rows.append({"record": record, **m})
            print(f"{record}: Se {m['se']:.4f} · +P {m['ppv']:.4f} · FN {m['fn']} · FP {m['fp']} "
                  f"· desfase mediano {m['median_offset']:+.0f} muestras")

    tp, fn, fp = (sum(r[k] for r in rows) for k in ("tp", "fn", "fp"))
    print(f"\nTotal ({len(rows)} registros): Se {tp / (tp + fn):.4f} · +P {tp / (tp + fp):.4f} "
          f"· FN {fn} · FP {fp}")


if __name__ == "__main__":
    main()
