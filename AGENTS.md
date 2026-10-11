# Proyecto: detección de anomalías en ECG con autoencoder (MIT-BIH)
Autoencoder Conv1D (PyTorch) entrenado solo con latidos normales (símbolo N). Error de reconstrucción alto = latido anómalo. Proyecto académico.

## Reglas duras
1. data/raw/ es de solo lectura: nunca modificar, mover ni borrar.
2. Nunca agregar a git: data/raw/, data/processed/, runs/, *.npz, *.pt (excepto models/frozen/), .venv/.
3. Cero rutas absolutas. Toda ruta sale de src/ecg/paths.py.
4. El split "test" (DS2) está prohibido en desarrollo. Solo src/ecg/evaluate/final_test.py lo usa, con --final, y lo ejecuta la persona guardiana del test.
5. Los splits por paciente se definen ÚNICAMENTE en src/ecg/config.py.
6. Los contratos de docs/contracts.md no se cambian sin actualizar ese archivo en el mismo PR (etiqueta contract-change).
7. Trabaja solo en las carpetas que indique la tarea. Si necesitas tocar algo fuera, detente y explícalo.
8. Hiperparámetros en configs/*.yaml, no en el código.
9. Toda aleatoriedad usa src/ecg/utils/seed.py.
10. Antes de terminar: `pytest -q` y `ruff check .` en verde. Reporta el resultado.
11. Si un número no coincide con lo esperado (conteos, formas, métricas), NO lo ajustes para que cuadre: detente y reporta la discrepancia.
12. No hagas git push ni merges. Commits locales pequeños, en español e imperativo.

## Convenciones
- Python 3.11. Nombres de código en inglés; comentarios y documentación en español.
- Cada entrenamiento crea runs/<run_id>/ con run_id = AAAAMMDD-HHMM_<autor>_<nombre>.
- Notebooks solo en notebooks/<persona>/, nunca importados desde src/.
- Dependencias nuevas: no las agregues; pídelas.
