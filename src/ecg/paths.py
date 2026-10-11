"""Definición canónica de rutas relativas al repositorio.

Regla 3 de AGENTS.md: Cero rutas absolutas. Toda ruta sale de src/ecg/paths.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raíz del repositorio calculada a partir de este archivo
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Directorio de datos crudos (sobreescribible con ECG_RAW_DIR)
_raw_env = os.environ.get("ECG_RAW_DIR")
RAW_DIR: Path = Path(_raw_env) if _raw_env else REPO_ROOT / "data" / "raw" / "mitbih_database"

# Directorios de datos derivados, ejecuciones y artefactos
PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"
RUNS_DIR: Path = REPO_ROOT / "runs"
RESULTS_DIR: Path = REPO_ROOT / "results"
FROZEN_DIR: Path = REPO_ROOT / "models" / "frozen"
CONFIGS_DIR: Path = REPO_ROOT / "configs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
REPORT_DIR: Path = REPO_ROOT / "report"
