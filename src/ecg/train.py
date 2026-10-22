"""Entrena el autoencoder con los latidos normales limpios de train (D-020 a D-022).

Uso: python -m ecg.train --config configs/train/baseline.yaml --author <autor>
Crea runs/<AAAAMMDD-HHMM_autor_nombre>/ con config.yaml, model.pt (pesos de la mejor época),
history.json (pérdidas por época) y meta.json (procedencia y resumen).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from ecg.io import load_beats
from ecg.models.autoencoder import ConvAutoencoder, count_parameters
from ecg.models.preprocessing import normal_mask, prepare_inputs
from ecg.paths import PROCESSED_DIR, REPO_ROOT, RUNS_DIR
from ecg.utils.seed import seed_everything


def load_config(path: Path | str) -> dict[str, Any]:
    """Lee una configuración de entrenamiento (YAML)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict[str, Any]) -> ConvAutoencoder:
    """Construye el autoencoder con la sección `model` de la configuración."""
    return ConvAutoencoder(**config["model"])


def make_run_id(author: str, name: str, now: datetime | None = None) -> str:
    """run_id = AAAAMMDD-HHMM_<autor>_<nombre> (convención de AGENTS.md)."""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M}_{author}_{name}"


def mean_loss(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """MSE medio de reconstrucción sobre `x`, sin actualizar pesos."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            total += nn.functional.mse_loss(model(xb), xb, reduction="sum").item()
    return total / x.numel()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return out.stdout.strip()


def _data_version(beats_path: Path) -> str | None:
    meta_path = beats_path.with_name("meta.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8")).get("data_version")


def train_autoencoder(
    config: dict[str, Any],
    author: str,
    beats_path: Path | str | None = None,
    runs_dir: Path | str = RUNS_DIR,
    verbose: bool = True,
) -> Path:
    """Entrena según `config` y devuelve la carpeta del run.

    Usa solo train y val (load_beats bloquea test). Se entrena con los N de train y se detiene
    cuando el error de los N de val no mejora en `patience` épocas (D-021); se guardan los
    pesos de la mejor época.
    """
    started = time.time()
    seed_everything(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    beats_path = Path(beats_path) if beats_path is not None else PROCESSED_DIR / "beats.npz"
    data_cfg, train_cfg = config["data"], config["training"]

    beats = load_beats(beats_path, splits=("train", "val"))
    train_mask = normal_mask(beats, "train", data_cfg["exclude_noisy"])
    val_mask = normal_mask(beats, "val", data_cfg["exclude_noisy"])
    x_train = torch.from_numpy(prepare_inputs(beats["X"][train_mask], data_cfg["normalization"]))
    x_val = torch.from_numpy(prepare_inputs(beats["X"][val_mask], data_cfg["normalization"]))
    x_train, x_val = x_train.unsqueeze(1), x_val.unsqueeze(1)

    run_dir = Path(runs_dir) / make_run_id(author, config["name"])
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    batch_size = train_cfg["batch_size"]
    if verbose:
        print(f"{run_dir.name}: {len(x_train):,} latidos de entrenamiento, "
              f"{len(x_val):,} de validación, {count_parameters(model):,} parámetros, {device}")

    history: list[dict[str, float]] = []
    best_loss, best_epoch, waited = float("inf"), 0, 0
    for epoch in range(1, train_cfg["max_epochs"] + 1):
        epoch_start = time.time()
        model.train()
        total = 0.0
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(x_train), batch_size):
            xb = x_train[order[start:start + batch_size]].to(device)
            loss = nn.functional.mse_loss(model(xb), xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(x_train)
        val_loss = mean_loss(model, x_val, batch_size, device)

        if val_loss < best_loss - train_cfg["min_delta"]:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            waited += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": round(time.time() - epoch_start, 2)})
        (run_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        if verbose:
            mark = " *" if best_epoch == epoch else ""
            print(f"época {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  "
                  f"{history[-1]['seconds']:.0f} s{mark}")
        if waited >= train_cfg["patience"]:
            break

    meta = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data_version": _data_version(beats_path),
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_train_noisy_excluded": int((normal_mask(beats, "train", False) & ~train_mask).sum()),
        "parameters": count_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "stopped_early": len(history) < train_cfg["max_epochs"],
        "seconds": round(time.time() - started, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"Mejor época {best_epoch} (val {best_loss:.5f}) · {meta['seconds']:.0f} s")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el autoencoder (D-020 a D-022).")
    parser.add_argument("--config", required=True, help="YAML en configs/train/")
    parser.add_argument("--author", required=True, help="autor para el run_id")
    args = parser.parse_args()
    train_autoencoder(load_config(args.config), args.author)


if __name__ == "__main__":
    main()
