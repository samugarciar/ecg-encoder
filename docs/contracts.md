# Contratos de Datos y Resultados

Este documento define formalmente las interfaces inmutables de datos entre fases del proyecto. Cualquier modificación a estos contratos requiere abrir un Pull Request con la etiqueta `contract-change` y actualizar este archivo.

---

## Contrato C1: Latidos Preprocesados
- **Archivos**: `data/processed/beats.npz` y `data/processed/meta.json`

### `beats.npz`
Diccionario de arrays de NumPy alineados a lo largo del eje `n` (número total de latidos extraídos):
- `X`: `float32`, forma `(n, 256)`. Señal de derivación MLII filtrada con pasabanda 0.5–40 Hz, expresada en milivoltios (mV), **SIN normalizar**.
- `symbol`: `<U1`, forma `(n,)`. Símbolo original del latido según anotación MIT-BIH (ej. `'N'`, `'L'`, `'R'`, `'V'`, `'A'`, etc.).
- `aami`: `<U1`, forma `(n,)`. Clase AAMI correspondiente en `{'N', 'S', 'V', 'F', 'Q'}`.
- `label`: `int8`, forma `(n,)`. Etiqueta binaria de anomalía del proyecto: `0` si `symbol == "N"`, de lo contrario `1`.
- `record`: `int16`, forma `(n,)`. Identificador del registro original (ej. `100`, `101`, `234`).
- `split`: `<U5`, forma `(n,)`. Subconjunto asignado al registro: `'train'`, `'val'` o `'test'`.
- `r_sample`: `int32`, forma `(n,)`. Índice de la muestra del pico R en el registro continuo de 360 Hz.
- `rr_prev`: `float32`, forma `(n,)`. Intervalo R-R previo en segundos (`NaN` para el primer latido del registro).
- `rr_next`: `float32`, forma `(n,)`. Intervalo R-R siguiente en segundos (`NaN` para el último latido del registro).
- `mlii_noisy`: `bool`, forma `(n,)`. `True` si el latido se encuentra dentro de un segmento marcado como ruidoso en MLII.

### `meta.json`
Metadatos del proceso de extracción:
- `data_version`: Versión del pipeline de preprocesamiento (ej. `"1.0.0"`).
- `git_commit`: Hash del commit con el que se generó.
- `fs`: Frecuencia de muestreo (360).
- `window`: Configuración de ventana `{"pre": 90, "post": 166, "total": 256}`.
- `filter`: Configuración del filtro Butterworth `{"low": 0.5, "high": 40.0, "order": 3}`.
- `counts`: Conteos de latidos desglosados por split y por clase AAMI (`split x aami`).
- `created_at`: Marca de tiempo ISO-8601 de creación.

---

## Contrato C2: Puntuaciones de Error del Modelo
- **Archivo**: `runs/<run_id>/scores_<split>.npz` (donde `<split>` está en `{'train', 'val', 'test'}`)

Diccionario de arrays de NumPy con el error de reconstrucción por latido:
- `idx`: `int64`, forma `(m,)`. Índice de fila correspondiente en `data/processed/beats.npz`.
- `error`: `float32`, forma `(m,)`. Error cuadrático medio (MSE) por latido, calculado sobre el latido normalizado exactamente igual que durante el entrenamiento.
- `label`: `int8`, forma `(m,)`. Etiqueta de verdad terreno (0 si `symbol == "N"`, si no 1).
- `symbol`: `<U1`, forma `(m,)`. Símbolo original del latido.
- `aami`: `<U1`, forma `(m,)`. Clase AAMI.
- `record`: `int16`, forma `(m,)`. Identificador del registro.
- `mlii_noisy`: `bool`, forma `(m,)`. Indicador de latido en segmento ruidoso.

---

## Contrato C3: Métricas y Evaluación del Experimento
- **Archivo**: `results/<run_id>__<split>.json`

Estructura JSON con los resultados de evaluación:
```json
{
  "run_id": "AAAAMMDD-HHMM_autor_nombre",
  "split": "val",
  "data_version": "1.0.0",
  "threshold": {
    "method": "percentile",
    "percentile": 95.0,
    "value": 0.0452,
    "fitted_on": "val"
  },
  "n": {
    "normal": 10500,
    "anomalous": 1200
  },
  "confusion": {
    "tp": 1050,
    "fp": 525,
    "fn": 150,
    "tn": 9975
  },
  "global": {
    "auc_roc": 0.9421,
    "auc_pr": 0.8874,
    "sensitivity": 0.8750,
    "specificity": 0.9500,
    "precision": 0.6667,
    "f1": 0.7568,
    "accuracy": 0.9423
  },
  "per_symbol_sensitivity": {
    "V": 0.92,
    "A": 0.81
  },
  "per_aami_sensitivity": {
    "S": 0.83,
    "V": 0.91,
    "F": 0.79,
    "Q": 0.88
  },
  "aami_view": {
    "auc_roc": 0.9510,
    "auc_pr": 0.8950,
    "sensitivity": 0.8820,
    "specificity": 0.9500,
    "precision": 0.6800,
    "f1": 0.7680,
    "accuracy": 0.9450
  },
  "clean_only": {
    "auc_roc": 0.9580,
    "auc_pr": 0.9020,
    "sensitivity": 0.8910,
    "specificity": 0.9550,
    "precision": 0.7100,
    "f1": 0.7900,
    "accuracy": 0.9510
  }
}
```
- `aami_view`: Mismas métricas que `global`, tratando los símbolos `L`, `R`, `e`, `j` como normales según la agrupación AAMI.
- `clean_only`: Mismas métricas que `global`, recalculadas tras excluir los latidos con `mlii_noisy == True`.
