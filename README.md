# Detección de Anomalías en ECG con Autoencoder (MIT-BIH)

Repositorio académico para la detección no supervisada de arritmias cardíacas en registros electrocardiográficos de la base de datos MIT-BIH Arrhythmia Database, utilizando una arquitectura de Autoencoder Convolucional 1D (Conv1D) en PyTorch.

---

## Estructura del Repositorio

```text
ecg-autoencoder/
├── AGENTS.md                          # Reglas duras y directrices para agentes y desarrolladores
├── README.md                          # Instrucciones del proyecto y tabla de roles
├── pyproject.toml                     # Configuración del paquete ecg, ruff y pytest
├── requirements.txt                   # Dependencias fijadas y multiplataforma
├── .gitignore                         # Exclusiones de git (datos, ejecuciones, modelos)
├── .github/
│   ├── CODEOWNERS                     # Asignación de módulos a responsables de área
│   ├── pull_request_template.md       # Plantilla de PR con checklist estricto
│   └── workflows/
│       └── ci.yml                     # Integración continua (Python 3.11, ruff, pytest)
├── configs/
│   └── train/                         # Configuraciones de hiperparámetros de entrenamiento
├── data/
│   ├── raw/
│   │   └── mitbih_database/           # 96 archivos crudos (48 registros .csv y .txt) [SOLO LECTURA]
│   ├── processed/                     # beats.npz y meta.json generados (Contrato C1)
│   ├── MANIFEST.sha256                # Checksums SHA-256 de los datos crudos
│   └── README.md                      # Documentación del origen y uso de datos
├── docs/
│   ├── contracts.md                   # Contratos de interfaces de datos (C1, C2, C3)
│   └── decisions.md                   # Registro de decisiones de arquitectura (ADR D-001 a D-019)
├── notebooks/                         # Notebooks exploratorios organizados por autor
├── models/
│   └── frozen/                        # Checkpoints congelados para evaluación final
├── results/                           # Métricas JSON (Contrato C3) y leaderboard.md
├── scripts/
│   ├── make_manifest.py               # Genera data/MANIFEST.sha256
│   ├── verify_raw_data.py             # Valida la integridad de data/raw/mitbih_database
│   ├── make_mock_beats.py             # Genera latidos sintéticos mock (C1)
│   ├── make_mock_scores.py            # Genera puntuaciones mock de error (C2)
│   └── aggregate_results.py           # Agrega resultados JSON a results/leaderboard.md
├── src/
│   └── ecg/
│       ├── __init__.py                # Inicializador del paquete ecg
│       ├── config.py                  # Constantes globales, splits inter-paciente y AAMI
│       ├── paths.py                   # Resolución canónica de rutas relativas
│       ├── io.py                      # Carga y validación estricta de contratos
│       ├── utils/
│       │   └── seed.py                # Control determinista de semillas pseudo-aleatorias
│       ├── data/                      # Módulo de pipeline y preprocesamiento
│       ├── models/                    # Módulo de arquitecturas y entrenamiento
│       └── evaluate/                  # Módulo de evaluación y reporte de métricas
└── tests/
    ├── test_config.py                 # Validación de splits y constantes de config.py
    ├── test_contracts.py              # Validación de contratos C1 y C2 sobre mocks
    └── test_io_guard.py               # Verificación de barrera de acceso al split test
```

---

## Instalación Paso a Paso

### 1. Clonar el repositorio y acceder a él
```bash
git clone <URL_DEL_REPOSITORIO>
cd ecg-autoencoder
```

### 2. Crear y activar el entorno virtual con Python 3.11
```bash
python3.11 -m venv .venv
source .venv/bin/activate
# En Windows: .venv\Scripts\activate
```

### 3. Instalar las dependencias y el paquete en modo editable
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Configurar la limpieza automática de notebooks
```bash
nbstripout --install
```

### 5. Disponer los datos crudos y verificar su integridad
Asegúrate de que los 96 archivos (48 archivos `.csv` y 48 `annotations.txt`) se encuentren en `data/raw/mitbih_database/`.
Luego, corre la verificación de sumas SHA-256:
```bash
python scripts/verify_raw_data.py
```
Si todo es correcto, verás: `OK: Los 96 archivos en RAW_DIR coinciden exactamente con el manifiesto.`

### 6. Ejecutar pruebas unitarias y linters
```bash
pytest -q
ruff check .
```

---

## Contexto de Datos

Para comprender y ejecutar el notebook del pipeline ([`notebooks/datos_mitbih_c1.ipynb`](notebooks/datos_mitbih_c1.ipynb)), todas las constantes físicas, criterios médicos y decisiones de diseño (`docs/decisions.md`) se centralizan en `src/ecg/config.py` (D-015) y se validan en `src/ecg/io.py`:

* **Señal física y conversión a mV (`FS`, `ADC_ZERO`, `ADC_GAIN`, `LEAD`)**:
  * Los 48 registros de **MIT-BIH** (D-001) duran 30 min a **`FS = 360` Hz** ($650,000$ muestras por archivo, $\Delta t \approx 2.78\text{ ms}$).
  * Los `.csv` crudos almacenan enteros del conversor A/D de 11 bits donde el cero eléctrico es **`ADC_ZERO = 1024`** y la ganancia es **`ADC_GAIN = 200`** unidades/mV: $\text{mV} = (x - 1024) / 200$.
  * Se extrae únicamente la derivación **`LEAD = "MLII"`** (D-004), buscándola por nombre de columna y no por posición (en el registro `114` está en el canal 1 en lugar del 0).
* **Filtrado y recorte de ventana (`BANDPASS`, `FILTER_ORDER`, `WIN_PRE`, `WIN_POST`, `WIN_LEN`)**:
  * **Filtro pasabanda Butterworth** (`BANDPASS = (0.5, 40.0)` Hz, `FILTER_ORDER = 3`, D-007): se aplica con fase cero (`sosfiltfilt`) sobre el registro continuo completo antes de segmentar, eliminando deriva de línea base ($<0.5\text{ Hz}$) y ruido muscular/red eléctrica ($>40\text{ Hz}$) sin desfasar el QRS ni generar transitorios en los cortes.
  * **Ventana asimétrica `[r - 90, r + 166)`** (`WIN_PRE = 90`, `WIN_POST = 166`, `WIN_LEN = 256`, D-006): cubre $250\text{ ms}$ antes y $461\text{ ms}$ después de la anotación del pico R ($\approx 711\text{ ms}$, desde la onda P hasta el fin de la onda T) en una potencia de 2 óptima para `Conv1D`. Los 39 latidos cuya ventana se sale de los extremos del registro se descartan por borde.
  * **Unidades en C1** (D-008): `beats.npz` guarda la ventana `X` en mV filtrados **sin normalizar**; el *z-score* por latido se aplica en la etapa de Modelo.
* **Partición inter-paciente (`EXCLUDED`, `DS1_TRAIN`, `DS1_VAL`, `DS2_TEST`)**:
  * Según el estándar médico **AAMI EC57**, se excluyen los 4 registros con marcapasos: **`EXCLUDED = [102, 104, 107, 217]`** (D-002).
  * Los **44 registros válidos** se dividen **por paciente** siguiendo a *de Chazal et al. (2004)* (D-003, D-013) para evitar fuga de morfología cardíaca entre splits: **`DS1_TRAIN`** (18 registros), **`DS1_VAL`** (4 registros: `116, 119, 209, 223`) y **`DS2_TEST`** (22 registros). `load_beats()` bloquea el acceso a `test` durante el desarrollo salvo que se indique `allow_test=True`.
* **Símbolos, clases AAMI y definición de normalidad (`AAMI_MAP`, `NORMAL_SYMBOLS`, `AAMI_NORMAL`)**:
  * Solo los **15 símbolos de `AAMI_MAP`** son complejos QRS clasificables y entran al Contrato C1, agrupados en las 5 superclases estándar (`N`, `S`, `V`, `F`, `Q`); quedan fuera $3,153$ anotaciones de eventos no-latido como cambios de ritmo `+`, calidad de señal `~`, ondas de aleteo ventricular `!` o artefactos `|` (D-014).
  * **Normalidad estricta** (D-005, D-018): en este proyecto `label = 0` se asigna **solo cuando `symbol == "N"`** (`NORMAL_SYMBOLS = {"N"}`, sin importar el ritmo activo en `+`). Otros símbolos de la clase AAMI N con morfología alterada (bloqueos de rama `L`, `R` y escapes `e`, `j`) llevan `label = 1` (anómalos), conservando `AAMI_NORMAL = {"N", "L", "R", "e", "j"}` para calcular la métrica comparativa `aami_view` en evaluación.
* **Intervalos RR y ruido en MLII (`rr_prev`, `rr_next`, `mlii_noisy`)**:
  * **`rr_prev` y `rr_next`** (D-017): tiempo en segundos hacia el latido AAMI anterior y siguiente dentro del mismo registro, calculado **antes** de descartar los 39 latidos de borde para preservar el intervalo fisiológico real de sus vecinos.
  * **`mlii_noisy`** (D-016): las marcas `~` actúan como un interruptor cuyo subtipo (`sub`) es una máscara de bits por canal (bit 0 = canal 0, bit 1 = canal 1, `-1` = ambos). Un latido recibe `mlii_noisy = True` si **cualquier muestra de su ventana `[r - 90, r + 166)`** toca un tramo ruidoso en el canal de `MLII`.

---

## Tabla de Roles y Responsabilidades

| Rol | Marcador CODEOWNERS | Rutas Asignadas | Responsabilidad Principal |
| :--- | :--- | :--- | :--- |
| **Integrador** | `@INTEGRADOR` | `src/ecg/config.py`<br>`requirements.txt`<br>`AGENTS.md`<br>`docs/contracts.md` | Gobierno de contratos, dependencias, CI y gestión de ramas/releases. |
| **Datos** | `@DATOS` | `src/ecg/data/`<br>`docs/contracts.md` | Ingesta, filtrado pasabanda, segmentación y generación de `beats.npz` (C1). |
| **Modelo** | `@MODELO` | `src/ecg/models/`<br>`src/ecg/train.py`<br>`src/ecg/score.py`<br>`configs/train/`<br>`docs/contracts.md` | Arquitectura Conv1D, bucle de entrenamiento en PyTorch y cálculo de error C2. |
| **Evaluación** | `@EVALUACION` | `src/ecg/evaluate/`<br>`docs/contracts.md` | Umbral en validación, métricas C3, matriz de confusión y leaderboard. |
| **Guardián del Test** | Designado manualmente | `src/ecg/evaluate/final_test.py`<br>`models/frozen/` | Ejecución única final sobre `DS2_TEST` con modelo congelado y bandera `--final`. |

---

## Reglas Críticas del Proyecto (Extracto de `AGENTS.md`)
1. **`data/raw/` es de solo lectura**: nunca modificar, mover ni borrar.
2. **Nunca versionar en git**: datos crudos ni procesados, checkpoints en `runs/`, archivos `*.npz` o `*.pt` (salvo `models/frozen/*.pt`).
3. **Cero rutas absolutas**: toda ruta debe ser importada desde `src/ecg/paths.py`.
4. **Split "test" (DS2) prohibido**: no debe usarse en desarrollo ni para selección de umbrales.
