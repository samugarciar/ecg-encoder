"""El preprocesamiento de la interfaz (web/js/dsp.js) da lo mismo que el pipeline de Python.

Corre el JavaScript con Node; si Node no está instalado, las pruebas se omiten.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from ecg.config import ADC_GAIN, ADC_ZERO, FS, WIN_POST, WIN_PRE
from ecg.models.preprocessing import zscore_beats
from ecg.paths import REPO_ROOT
from scripts.export_web_model import preprocessing_spec
from scripts.make_demo_ecg import save_demo_ecg, synthetic_ecg

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="requiere Node.js")
WEB_JS = REPO_ROOT / "web" / "js"

RUNNER = """
import { readFileSync, writeFileSync } from "node:fs";
import * as dsp from "./dsp.js";
const [, , inPath, outPath] = process.argv;
const calls = JSON.parse(readFileSync(inPath, "utf8"));
const out = {};
for (const [name, call] of Object.entries(calls)) {
  try {
    // {"f64": [...]} llega como Float64Array; los typed arrays vuelven como listas
    const args = call.args.map((a) => (a && a.f64 ? Float64Array.from(a.f64) : a));
    const value = dsp[call.fn](...args);
    out[name] = JSON.parse(JSON.stringify(value, (k, v) =>
      (ArrayBuffer.isView(v) ? Array.from(v) : v)));
  } catch (e) {
    out[name] = { error: e.message };
  }
}
writeFileSync(outPath, JSON.stringify(out));
"""


def _run_js(tmp_path: Path, calls: dict) -> dict:
    """Ejecuta funciones de dsp.js con Node y devuelve sus resultados."""
    for js in WEB_JS.glob("*.js"):
        shutil.copy(js, tmp_path / js.name)
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (tmp_path / "runner.js").write_text(RUNNER, encoding="utf-8")
    in_path, out_path = tmp_path / "in.json", tmp_path / "out.json"
    in_path.write_text(json.dumps(calls), encoding="utf-8")
    subprocess.run([NODE, "runner.js", str(in_path), str(out_path)], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def _f64(x: np.ndarray) -> dict:
    return {"f64": np.asarray(x, dtype=np.float64).tolist()}


def test_sosfiltfilt_matches_scipy(tmp_path: Path) -> None:
    prep = preprocessing_spec()
    sos = np.asarray(prep["filter"]["sos"])
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=4000)) * 0.01 + rng.normal(size=4000)
    short = np.ones(prep["filter"]["padlen"])
    out = _run_js(tmp_path, {
        "long": {"fn": "sosfiltfilt", "args": [prep["filter"], _f64(x)]},
        "short": {"fn": "sosfiltfilt", "args": [prep["filter"], _f64(short)]},
    })
    expected = signal.sosfiltfilt(sos, x)
    assert np.max(np.abs(np.asarray(out["long"]) - expected)) < 1e-9
    assert "demasiado corta" in out["short"]["error"]


def test_windows_and_zscore_match_python(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=3000)
    r = np.array([50, 400, 1200, 2900])  # el primero y el último no caben
    out = _run_js(tmp_path, {
        "win": {"fn": "extractWindows", "args": [_f64(y), r.tolist(), WIN_PRE, WIN_POST]},
    })
    assert out["win"]["keep"] == [1, 2]
    windows = np.asarray(out["win"]["windows"], dtype=np.float32).reshape(2, -1)
    idx = r[[1, 2], None] + np.arange(-WIN_PRE, WIN_POST)
    expected = y[idx].astype(np.float32)  # C1 guarda X en float32
    assert np.array_equal(windows, expected)

    z = _run_js(tmp_path, {"z": {"fn": "zscoreRows", "args": [windows.ravel().tolist(), 256]}})
    assert np.allclose(np.asarray(z["z"]).reshape(2, -1), zscore_beats(expected), atol=1e-5)


def test_mitbih_csv_and_annotations_are_parsed(tmp_path: Path) -> None:
    csv_path, ann_path = save_demo_ecg(tmp_path / "samples", seconds=10.0)
    csv_text = csv_path.read_text(encoding="utf-8")
    ann_text = ann_path.read_text(encoding="utf-8")
    prep = preprocessing_spec()
    out = _run_js(tmp_path, {
        "csv": {"fn": "parseSignalCsv", "args": [csv_text]},
        "ann": {"fn": "parseAnnotations", "args": [ann_text, prep["aami_map"]]},
        "semicolon": {"fn": "parseSignalCsv", "args": ["tiempo;mV\n0;0,5\n0,0025;-1,25\n"]},
    })
    assert out["csv"]["columns"] == ["sample #", "MLII", "V5"]
    mlii, r_samples, symbols = synthetic_ecg(10.0)
    adc = np.asarray(out["csv"]["values"][1])
    assert np.array_equal(adc, np.round(mlii * ADC_GAIN + ADC_ZERO))
    assert out["ann"]["samples"] == r_samples.tolist()  # el "+" del ritmo no es un latido
    assert out["ann"]["symbols"] == symbols
    assert out["semicolon"]["values"] == [[0.0, 0.0025], [0.5, -1.25]]


def test_r_detector_finds_synthetic_beats(tmp_path: Path) -> None:
    """En el ECG sintético (con deriva y ruido de red) encuentra todos los R, a <= 10 ms."""
    mlii, r_true, _ = synthetic_ecg(60.0)
    prep = preprocessing_spec()
    y = signal.sosfiltfilt(np.asarray(prep["filter"]["sos"]), mlii)
    out = _run_js(tmp_path, {"r": {"fn": "detectRPeaks", "args": [_f64(y), FS]}})
    r_found = np.asarray(out["r"])
    tolerance = int(0.15 * FS)  # ventana de emparejamiento estándar de 150 ms
    nearest = np.abs(r_true[:, None] - r_found[None, :]).min(axis=1)
    assert np.all(nearest <= tolerance), "faltan latidos"
    assert len(r_found) == len(r_true), "sobran latidos"
    assert np.max(nearest) <= int(0.010 * FS)
