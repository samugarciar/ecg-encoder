"""Verifica la integridad de los datos crudos en RAW_DIR contra data/MANIFEST.sha256."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from ecg.paths import RAW_DIR, REPO_ROOT


def compute_sha256(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    """Compara RAW_DIR con data/MANIFEST.sha256 e informa discrepancias."""
    manifest_path = REPO_ROOT / "data" / "MANIFEST.sha256"
    if not manifest_path.exists():
        print(f"Error: No se encontró el manifiesto en: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if not RAW_DIR.exists():
        print(f"Error: No se encontró el directorio RAW_DIR en: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    expected_hashes: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                sha, filename = parts[0], parts[1].strip()
                expected_hashes[filename] = sha

    actual_files = {f.name for f in RAW_DIR.iterdir() if f.is_file() and not f.name.startswith(".")}

    missing_files = sorted(set(expected_hashes.keys()) - actual_files)
    extra_files = sorted(actual_files - set(expected_hashes.keys()))
    mismatched_hashes: list[tuple[str, str, str]] = []

    for filename in sorted(set(expected_hashes.keys()) & actual_files):
        actual_sha = compute_sha256(RAW_DIR / filename)
        expected_sha = expected_hashes[filename]
        if actual_sha != expected_sha:
            mismatched_hashes.append((filename, expected_sha, actual_sha))

    if missing_files or extra_files or mismatched_hashes:
        print("FALLO en la verificación de datos crudos:")
        if missing_files:
            print(f"  - Archivos faltantes ({len(missing_files)}): {missing_files}")
        if extra_files:
            print(f"  - Archivos no registrados ({len(extra_files)}): {extra_files}")
        if mismatched_hashes:
            print(f"  - Discrepancias de hash ({len(mismatched_hashes)}):")
            for fn, exp, act in mismatched_hashes:
                print(f"      {fn}: esperado={exp[:8]}... obtenido={act[:8]}...")
        sys.exit(1)

    msg = (
        f"OK: Los {len(expected_hashes)} archivos en RAW_DIR "
        "coinciden exactamente con el manifiesto."
    )
    print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()
