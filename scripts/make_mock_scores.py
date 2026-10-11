"""Generación de puntuaciones de error sintéticas mock que cumplen el Contrato C2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecg.paths import PROCESSED_DIR, RUNS_DIR
from ecg.utils.seed import seed_everything

try:
    from scripts.make_mock_beats import generate_mock_beats
except ModuleNotFoundError:
    from make_mock_beats import generate_mock_beats


def generate_mock_scores(
    beats: dict[str, np.ndarray],
    split: str = "val",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Genera un diccionario de arrays de evaluación según el Contrato C2.

    Utiliza una distribución lognormal para el error cuadrático medio:
    - Latidos normales (label=0): errores centrados en valores bajos (~0.035).
    - Latidos anómalos (label=1): errores centrados en valores altos (~0.220).
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    split_arr = np.asarray(beats["split"]).astype(str)
    split_indices = np.where(split_arr == split)[0]

    if len(split_indices) == 0:
        raise ValueError(
            f"No se encontraron latidos con split='{split}' en los datos proporcionados"
        )

    m = len(split_indices)
    labels = np.asarray(beats["label"])[split_indices].astype(np.int8)
    symbols = np.asarray(beats["symbol"])[split_indices].astype("<U1")
    aami_arr = np.asarray(beats["aami"])[split_indices].astype("<U1")
    records = np.asarray(beats["record"])[split_indices].astype(np.int16)
    noisy = np.asarray(beats["mlii_noisy"])[split_indices].astype(np.bool_)

    # Errores lognormales diferenciados
    errors = np.zeros(m, dtype=np.float32)
    norm_mask = labels == 0
    anom_mask = labels == 1

    # Normales: media logarítmica log(0.035), sigma 0.30
    errors[norm_mask] = rng.lognormal(mean=np.log(0.035), sigma=0.30, size=np.sum(norm_mask))
    # Anómalos: media logarítmica log(0.220), sigma 0.40
    errors[anom_mask] = rng.lognormal(mean=np.log(0.220), sigma=0.40, size=np.sum(anom_mask))

    return {
        "idx": split_indices.astype(np.int64),
        "error": errors.astype(np.float32),
        "label": labels,
        "symbol": symbols,
        "aami": aami_arr,
        "record": records,
        "mlii_noisy": noisy,
    }


def save_mock_scores(
    run_id: str = "mock",
    split: str = "val",
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """Guarda las puntuaciones sintéticas en runs/<run_id>/scores_<split>.npz."""
    beats_path = PROCESSED_DIR / "beats_mock.npz"
    if beats_path.exists():
        with np.load(beats_path, allow_pickle=False) as data:
            beats = {k: np.asarray(data[k]) for k in data.files}
    else:
        # Generar latidos sintéticos en memoria
        beats, _ = generate_mock_beats(n_beats=1500, seed=seed)

    scores = generate_mock_scores(beats, split=split, seed=seed)

    target_dir = output_dir if output_dir is not None else RUNS_DIR / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"scores_{split}.npz"
    np.savez_compressed(out_file, **scores)
    return out_file


def main() -> None:
    """Punto de entrada para generar runs/mock/scores_val.npz."""
    out_path = save_mock_scores(run_id="mock", split="val", seed=42)
    print(f"Mock C2 generado exitosamente en: {out_path}")


if __name__ == "__main__":
    main()
