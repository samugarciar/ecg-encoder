"""Genera el manifiesto data/MANIFEST.sha256 con los hashes de los archivos en RAW_DIR."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from ecg.paths import RAW_DIR, REPO_ROOT


def compute_sha256(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en bloques de 64 KB."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    """Escanea RAW_DIR y genera el manifiesto ordenado."""
    if not RAW_DIR.exists():
        print(f"Error: No existe el directorio RAW_DIR: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted([f for f in RAW_DIR.iterdir() if f.is_file() and not f.name.startswith(".")])
    manifest_path = REPO_ROOT / "data" / "MANIFEST.sha256"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as out_file:
        for file in files:
            sha = compute_sha256(file)
            out_file.write(f"{sha}  {file.name}\n")

    print(f"Manifiesto generado exitosamente: {len(files)} archivos registrados en {manifest_path}")


if __name__ == "__main__":
    main()
