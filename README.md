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
