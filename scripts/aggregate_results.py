"""Agrega y formatea los resultados de evaluación de results/*.json en results/leaderboard.md."""

from __future__ import annotations

import json
from typing import Any

from ecg.paths import RESULTS_DIR


def format_metric(val: Any) -> str:
    """Formatea métricas flotantes a 4 decimales o guión si no existe."""
    if val is None:
        return "-"
    if isinstance(val, int | float):
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    """Escanea results/*.json y genera la tabla comparativa en results/leaderboard.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(RESULTS_DIR.glob("*.json"))

    leaderboard_file = RESULTS_DIR / "leaderboard.md"

    if not json_files:
        content = (
            "# Leaderboard de Experimentos\n\n"
            "No se han encontrado archivos de resultados en `results/*.json`.\n"
        )
        with open(leaderboard_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Aviso: No hay resultados en {RESULTS_DIR}. Archivo base creado.")
        return

    rows: list[dict[str, Any]] = []
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            rows.append(data)
        except Exception as e:
            print(f"Advertencia: No se pudo leer {jf.name}: {e}")

    # Ordenar por AUC-ROC descendente si está disponible
    def sort_key(item: dict[str, Any]) -> float:
        glob = item.get("global", {})
        return float(glob.get("auc_roc", 0.0))

    rows.sort(key=sort_key, reverse=True)

    hdr1 = (
        "| Run ID | Split | AUC-ROC | AUC-PR | F1 | Precision | "
        "Sensibilidad | Especificidad | Acc | Sens. S | Sens. V | Sens. F | Sens. Q |"
    )
    hdr2 = (
        "| :--- | :---: | :---: | :---: | :---: | :---: | "
        ":---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    lines: list[str] = [
        "# Tabla Comparativa de Rendimiento (Leaderboard)",
        "",
        hdr1,
        hdr2,
    ]

    for r in rows:
        run_id = r.get("run_id", "desconocido")
        split = r.get("split", "-")
        g = r.get("global", {})
        aami_s = r.get("per_aami_sensitivity", {})

        row_str = (
            f"| `{run_id}` "
            f"| `{split}` "
            f"| {format_metric(g.get('auc_roc'))} "
            f"| {format_metric(g.get('auc_pr'))} "
            f"| {format_metric(g.get('f1'))} "
            f"| {format_metric(g.get('precision'))} "
            f"| {format_metric(g.get('sensitivity'))} "
            f"| {format_metric(g.get('specificity'))} "
            f"| {format_metric(g.get('accuracy'))} "
            f"| {format_metric(aami_s.get('S'))} "
            f"| {format_metric(aami_s.get('V'))} "
            f"| {format_metric(aami_s.get('F'))} "
            f"| {format_metric(aami_s.get('Q'))} |"
        )
        lines.append(row_str)

    lines.append("")
    with open(leaderboard_file, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))

    print(f"Leaderboard actualizado exitosamente en {leaderboard_file} con {len(rows)} resultados.")


if __name__ == "__main__":
    main()
