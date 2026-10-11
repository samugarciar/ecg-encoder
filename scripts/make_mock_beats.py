"""Generación de latidos sintéticos mock que cumplen rigurosamente el Contrato C1."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ecg.config import (
    AAMI_MAP,
    DS1_TRAIN,
    DS1_VAL,
    DS2_TEST,
    FILTER_ORDER,
    FS,
    WIN_LEN,
    WIN_POST,
    WIN_PRE,
)
from ecg.paths import PROCESSED_DIR
from ecg.utils.seed import seed_everything


def _synthetic_beat(
    t: np.ndarray,
    symbol: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera forma de onda de ECG sintética (256 muestras) combinando gaussianas."""
    # El pico R se sitúa en la muestra WIN_PRE (90)
    r_idx = WIN_PRE

    if symbol == "N":
        # Latido normal: onda P, complejo QRS estrecho, onda T normal
        p_wave = 0.15 * np.exp(-0.5 * ((t - (r_idx - 45)) / 7.0) ** 2)
        q_wave = -0.15 * np.exp(-0.5 * ((t - (r_idx - 7)) / 3.0) ** 2)
        r_wave = 1.25 * np.exp(-0.5 * ((t - r_idx) / 4.5) ** 2)
        s_wave = -0.25 * np.exp(-0.5 * ((t - (r_idx + 8)) / 3.5) ** 2)
        t_wave = 0.32 * np.exp(-0.5 * ((t - (r_idx + 68)) / 16.0) ** 2)
    elif symbol in {"V", "E"}:
        # Ectópico ventricular: sin onda P clara, QRS ancho y mellado, onda T invertida
        p_wave = 0.02 * np.exp(-0.5 * ((t - (r_idx - 40)) / 10.0) ** 2)
        q_wave = -0.30 * np.exp(-0.5 * ((t - (r_idx - 14)) / 6.0) ** 2)
        r_wave = 1.45 * np.exp(-0.5 * ((t - r_idx) / 12.0) ** 2)
        s_wave = -0.55 * np.exp(-0.5 * ((t - (r_idx + 18)) / 8.0) ** 2)
        t_wave = -0.40 * np.exp(-0.5 * ((t - (r_idx + 75)) / 22.0) ** 2)
    elif symbol in {"A", "a", "J", "S"}:
        # Ectópico supraventricular: onda P prematura o anómala, QRS normal
        p_wave = 0.22 * np.exp(-0.5 * ((t - (r_idx - 32)) / 5.0) ** 2)
        q_wave = -0.12 * np.exp(-0.5 * ((t - (r_idx - 7)) / 3.0) ** 2)
        r_wave = 1.20 * np.exp(-0.5 * ((t - r_idx) / 4.5) ** 2)
        s_wave = -0.22 * np.exp(-0.5 * ((t - (r_idx + 8)) / 3.5) ** 2)
        t_wave = 0.28 * np.exp(-0.5 * ((t - (r_idx + 64)) / 15.0) ** 2)
    elif symbol in {"L", "R"}:
        # Bloqueo de rama: QRS moderadamente ensanchado
        p_wave = 0.14 * np.exp(-0.5 * ((t - (r_idx - 44)) / 7.0) ** 2)
        q_wave = -0.10 * np.exp(-0.5 * ((t - (r_idx - 10)) / 4.0) ** 2)
        r_wave = 1.15 * np.exp(-0.5 * ((t - r_idx) / 8.5) ** 2)
        s_wave = -0.40 * np.exp(-0.5 * ((t - (r_idx + 12)) / 5.0) ** 2)
        t_wave = 0.30 * np.exp(-0.5 * ((t - (r_idx + 70)) / 18.0) ** 2)
    else:
        # Fusión / Desconocido (F, Q)
        p_wave = 0.08 * np.exp(-0.5 * ((t - (r_idx - 38)) / 8.0) ** 2)
        q_wave = -0.20 * np.exp(-0.5 * ((t - (r_idx - 9)) / 4.0) ** 2)
        r_wave = 0.95 * np.exp(-0.5 * ((t - r_idx) / 10.0) ** 2)
        s_wave = -0.30 * np.exp(-0.5 * ((t - (r_idx + 14)) / 6.0) ** 2)
        t_wave = 0.18 * np.exp(-0.5 * ((t - (r_idx + 66)) / 20.0) ** 2)

    signal = p_wave + q_wave + r_wave + s_wave + t_wave
    noise = rng.normal(loc=0.0, scale=0.025, size=t.shape)
    return (signal + noise).astype(np.float32)


def generate_mock_beats(
    n_beats: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Genera arrays y metadatos sintéticos que cumplen rigurosamente el Contrato C1."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    # Distribución de latidos por split (~60% train, ~15% val, ~25% test)
    records_split_map: dict[int, str] = {}
    for r in DS1_TRAIN:
        records_split_map[r] = "train"
    for r in DS1_VAL:
        records_split_map[r] = "val"
    for r in DS2_TEST:
        records_split_map[r] = "test"

    all_records = list(records_split_map.keys())

    # Generar latidos por paciente para calcular intervalos R-R creíbles
    beats_per_record = max(10, n_beats // len(all_records))

    x_list: list[np.ndarray] = []
    symbol_list: list[str] = []
    aami_list: list[str] = []
    label_list: list[int] = []
    record_list: list[int] = []
    split_list: list[str] = []
    r_sample_list: list[int] = []
    rr_prev_list: list[float] = []
    rr_next_list: list[float] = []
    noisy_list: list[bool] = []

    t_axis = np.arange(WIN_LEN, dtype=np.float32)

    # Probabilidades de símbolos: ~82% normales, ~18% anormales variados
    anomaly_symbols = ["V", "A", "L", "R", "F", "Q"]
    anomaly_probs = [0.45, 0.25, 0.15, 0.08, 0.04, 0.03]

    for rec in all_records:
        rec_split = records_split_map[rec]
        # Determinar cuántos latidos para este registro
        k = beats_per_record + rng.integers(-5, 6)
        k = max(k, 5)

        # Generar picos R con frecuencia promedio de ~72 lpm (RR ~ 0.83 s = 300 muestras)
        r_positions: list[int] = []
        cur_sample = rng.integers(150, 400)
        for _ in range(k):
            r_positions.append(cur_sample)
            step = int(rng.normal(loc=0.83 * FS, scale=0.06 * FS))
            cur_sample += max(step, 180)

        for i, r_pos in enumerate(r_positions):
            # Determinar si es normal o anómalo
            # Si es split train, garantizar 100% normales para ser fiel al protocolo
            if rec_split == "train":
                sym = "N"
            else:
                if rng.random() < 0.20:
                    sym = rng.choice(anomaly_symbols, p=anomaly_probs)
                else:
                    sym = "N"

            aami = AAMI_MAP[sym]
            lbl = 0 if sym == "N" else 1

            # Calcular intervalos RR en segundos
            if i == 0:
                rr_p = np.nan
            else:
                rr_p = float((r_pos - r_positions[i - 1]) / FS)

            if i == len(r_positions) - 1:
                rr_n = np.nan
            else:
                rr_n = float((r_positions[i + 1] - r_pos) / FS)

            # Ruido en señal (aproximadamente un 3% de los latidos)
            is_noisy = bool(rng.random() < 0.03)

            wave = _synthetic_beat(t_axis, sym, rng)

            x_list.append(wave)
            symbol_list.append(sym)
            aami_list.append(aami)
            label_list.append(lbl)
            record_list.append(rec)
            split_list.append(rec_split)
            r_sample_list.append(r_pos)
            rr_prev_list.append(rr_p)
            rr_next_list.append(rr_n)
            noisy_list.append(is_noisy)

    data_dict: dict[str, np.ndarray] = {
        "X": np.array(x_list, dtype=np.float32),
        "symbol": np.array(symbol_list, dtype="<U1"),
        "aami": np.array(aami_list, dtype="<U1"),
        "label": np.array(label_list, dtype=np.int8),
        "record": np.array(record_list, dtype=np.int16),
        "split": np.array(split_list, dtype="<U5"),
        "r_sample": np.array(r_sample_list, dtype=np.int32),
        "rr_prev": np.array(rr_prev_list, dtype=np.float32),
        "rr_next": np.array(rr_next_list, dtype=np.float32),
        "mlii_noisy": np.array(noisy_list, dtype=np.bool_),
    }

    # Conteos para meta.json: split x aami
    counts: dict[str, dict[str, int]] = {}
    for s in ["train", "val", "test"]:
        counts[s] = {c: 0 for c in ["N", "S", "V", "F", "Q"]}

    for s, c in zip(split_list, aami_list, strict=True):
        counts[s][c] += 1

    meta_dict: dict[str, Any] = {
        "data_version": "0.1.0-mock",
        "git_commit": "synthetic",
        "fs": FS,
        "window": {"pre": WIN_PRE, "post": WIN_POST, "total": WIN_LEN},
        "filter": {"low": 0.5, "high": 40.0, "order": FILTER_ORDER},
        "counts": counts,
        "created_at": datetime.now(UTC).isoformat(),
    }

    return data_dict, meta_dict


def save_mock_beats(
    target_dir: Path | None = None,
    filename_prefix: str = "beats_mock",
    n_beats: int = 2000,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Genera y guarda en disco los artefactos sintéticos beats_mock.npz y meta_mock.json."""
    out_dir = target_dir if target_dir is not None else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data, meta = generate_mock_beats(n_beats=n_beats, seed=seed)

    npz_path = out_dir / f"{filename_prefix}.npz"
    meta_path = out_dir / f"{filename_prefix.replace('beats', 'meta')}.json"

    np.savez_compressed(npz_path, **data)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return npz_path, meta_path


def main() -> None:
    """Ejecuta la generación por defecto y confirma la creación de los mocks."""
    npz_path, meta_path = save_mock_beats(PROCESSED_DIR, "beats_mock", n_beats=2000, seed=42)
    print(f"Mock C1 generado con éxito:\n  - {npz_path}\n  - {meta_path}")


if __name__ == "__main__":
    main()
