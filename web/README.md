# Interfaz web para médicos

Página estática que analiza un ECG en el navegador con el autoencoder del proyecto y señala los
latidos que el modelo reconstruye mal (**atípicos**) para que un profesional los revise. Se
despliega en Vercel sin servidor: la señal del paciente no sale de su computador.

> **Estado actual: funciona de punta a punta, pero con un modelo de DEMOSTRACIÓN** entrenado con
> latidos sintéticos. La página lo avisa con una franja roja. Para conectar el modelo real, ver
> [Conectar el modelo real](#conectar-el-modelo-real).

Prueba rápida en local:

```bash
python scripts/serve_web.py          # http://localhost:8000
# http://localhost:8000/?ejemplo     abre la página con el ECG sintético ya analizado
```

---

## Qué se hizo

### La interfaz (`web/`)

| Archivo | Qué hace |
| :--- | :--- |
| `index.html`, `styles.css` | La página: carga del archivo, resultados y “Cómo funciona y qué tener en cuenta”. Modo claro y oscuro, versión móvil e impresión. |
| `js/dsp.js` | Procesamiento de señal, sin DOM: lectura del CSV y de las anotaciones, conversión a mV, remuestreo, filtro `sosfiltfilt`, detector de picos R, ventanas, z-score y MSE. |
| `js/analysis.js` | Encadena todo: de la señal cruda al error de reconstrucción y la decisión por latido. |
| `js/model.js` | Carga `model/` y corre el ONNX con onnxruntime-web 1.20.1 (WebAssembly, desde el CDN de jsDelivr). |
| `js/charts.js`, `js/app.js` | Gráficos en canvas y manejo de la página. |
| `model/model.onnx` | El autoencoder exportado (unos 1,2 MB). |
| `model/model.json` | Run, umbral, normalización y métricas de validación que se muestran al médico. |
| `model/preprocessing.json` | fs, conversión A/D, ventana y coeficientes del filtro, **generados desde `src/ecg/config.py`** (D-015). |
| `samples/` | ECG continuo **sintético** de 60 s con sus anotaciones, para el botón de ejemplo. |
| `vercel.json` | Despliegue estático y cabeceras de seguridad (CSP que solo permite el CDN de onnxruntime). |

**Qué ve el médico**
1. Carga un CSV con la señal (y, opcionalmente, las anotaciones de los latidos). La derivación, la
   frecuencia y las unidades se detectan solas y se pueden corregir antes de analizar.
2. Resumen: latidos atípicos (n y %), latidos analizados, FC mediana, duración y de dónde salieron
   los picos R.
3. Error de cada latido en el tiempo (veces el umbral). Al hacer clic en un latido, se ve en el
   trazado filtrado (cuadrícula de 0,2 s × 0,5 mV) y superpuesto con su reconstrucción.
4. Tabla de latidos ordenable, con navegación entre atípicos, descarga en CSV e impresión.
5. Si cargó anotaciones: cuántos latidos de cada símbolo se marcaron (sensibilidad y especificidad
   frente al cardiólogo).
6. El método, las limitaciones y las métricas del modelo, en lenguaje claro.

**Formatos que acepta**
- Señal: CSV con comas, punto y coma o tabulaciones, con o sin encabezado. Si hay una columna
  `MLII` se usa esa. Unidades: conversor A/D de MIT-BIH (`(x − 1024) / 200`), mV o µV. Con otra
  frecuencia se remuestrea a 360 Hz y la página lo advierte.
- Anotaciones (opcionales): `annotations.txt` de MIT-BIH o un CSV `muestra,símbolo`. Si no las
  hay, los R se ubican con un detector tipo Pan-Tompkins, y la página advierte que no está validado.

### Python (`scripts/`, `configs/`, `tests/`)

| Archivo | Qué hace |
| :--- | :--- |
| `scripts/export_web_model.py` | `--run runs/<run_id>` exporta un run real a `web/model/`; `--demo` entrena y exporta el modelo de demostración. Toma el umbral de `results/<run_id>__val.json` (C3) o de `scores_val.npz`. Nunca usa test. |
| `scripts/make_demo_ecg.py` | Genera `web/samples/` con las formas de latido de `make_mock_beats.py`. |
| `scripts/serve_web.py` | Servidor local con los mismos tipos MIME y cabeceras que Vercel. |
| `scripts/validate_r_detector.py` | Mide el detector de picos R contra las anotaciones de MIT-BIH (solo DS1). |
| `configs/train/demo_web.yaml` | Hiperparámetros del modelo de demostración (misma arquitectura que el baseline). |
| `tests/test_web_export.py` | El ONNX reconstruye igual que PyTorch, el umbral sale de val, `preprocessing.json` está sincronizado con `config.py`, y el detector rechaza registros de test. |
| `tests/test_web_dsp.py` | Corre `dsp.js` con Node y lo compara con scipy y numpy. Si no hay Node, se omite (en GitHub Actions sí hay). |

Dependencia nueva: `onnx==1.17.0` en `requirements.txt` (aprobada), para verificar la exportación.

### Cómo se verificó

- **ONNX contra PyTorch:** diferencia máxima de 2,4e-6 en la reconstrucción. El ONNX solo usa
  operadores básicos: el BatchNorm queda dentro de las Conv y no hay `Resize`.
- **JavaScript contra Python:** el filtro coincide con `scipy.signal.sosfiltfilt` con un error menor
  a 1e-9. Las ventanas son idénticas y el z-score coincide a 1e-5.
- **De punta a punta:** con el ECG de ejemplo, onnxruntime-web en Node dio el mismo error por
  latido que `ecg.score` (diferencia máxima de 1,2e-7) y las mismas 72 decisiones.
- **En navegador:** Brave sin interfaz, con la CSP de `vercel.json` activa, sin errores de consola.
  Se probaron el modo claro, el oscuro y un ancho de 390 px.
- **Caso genérico:** una sola columna en mV a 250 Hz, sin anotaciones. Se remuestreó, el detector
  encontró los 72 latidos y salieron los mismos 14 atípicos.
- `pytest -q`: 53 aprobadas y 1 omitida (GPU). `ruff check .`: sin errores.

---

## Conectar el modelo real

El repositorio no tiene pesos entrenados: `runs/` no se sube a git (regla 2). El único run
reportado, `20261104-0252_andres_baseline_gpu` (AUC-ROC 0,887 y sensibilidad 56 % en val), está en el
computador de Andrés. Para conectarlo:

1. Copiar `runs/20261104-0252_andres_baseline_gpu/` completo (`config.yaml`, `model.pt`,
   `meta.json`) a `runs/` en la máquina donde se va a exportar.
2. Exportar. El umbral se lee de `results/20261104-0252_andres_baseline_gpu__val.json`, que ya está
   en el repositorio:
   ```bash
   python -m scripts.export_web_model --run runs/20261104-0252_andres_baseline_gpu
   ```
3. `pytest -q` (revisa que `web/model/` sea coherente) y hacer commit de `web/model/`. La franja de
   demostración desaparece sola, porque `model.json` queda con `"is_demo": false`.

El `model.pt` no se sube a git: solo se publica `model.onnx`. Para volver al modelo de
demostración: `python -m scripts.export_web_model --demo`.

## Desplegar en Vercel

La carpeta `web/` es un sitio estático: no hay build ni funciones de servidor. PyTorch no cabría en
una función de Vercel (límite de unos 250 MB), por eso la inferencia corre en el navegador.

1. En vercel.com: **Add New → Project** e importar el repositorio de GitHub.
2. **Root Directory: `web`** (importante: así Vercel no intenta instalar los paquetes de Python del
   repositorio). Framework Preset: **Other**. Build e Install quedan vacíos (`vercel.json` ya lo
   indica).
3. **Deploy.** Cada push a `main` vuelve a desplegar.

Con la CLI, desde `web/`: `npx vercel` para una vista previa y `npx vercel --prod` para producción.

Al terminar, revisar que `https://<proyecto>.vercel.app/?ejemplo` muestre el análisis del ECG de
ejemplo.

---

## Qué falta por hacer

**Antes de mostrárselo a médicos**
- [ ] **Conectar el modelo real** (sección anterior). Mientras tanto, la página muestra un modelo
      de demostración sin valor clínico.
- [ ] **Validar el detector de picos R con MIT-BIH:** `python -m scripts.validate_r_detector`
      (requiere `data/raw/` y Node). Solo se probó con el ECG sintético. Lo esperable en la
      literatura es Se y +P por encima de 99 %. Si da menos, ajustarlo o pedir siempre anotaciones.
- [ ] **Hacer el despliegue en Vercel** (requiere la cuenta del equipo). Nadie lo ha hecho todavía.
- [ ] Revisar con alguien del área de la salud los textos de la página, sobre todo las
      limitaciones y la palabra “atípico”.

**Del proyecto, encontradas en la revisión**
- [ ] El modelo solo se ha medido en val. `src/ecg/evaluate/final_test.py` todavía no existe, así
      que no hay métricas en el conjunto de test (DS2) para mostrar.
- [ ] `src/ecg/data/` sigue vacío: C1 se genera desde el notebook (D-019 pendiente).
- [ ] Agregar `web/` y los scripts nuevos a `.github/CODEOWNERS`.

**Mejoras posibles**
- Hospedar onnxruntime-web en `web/vendor/` en vez del CDN (unos 10 MB en git), para redes
  hospitalarias que bloqueen jsDelivr.
- Leer directamente archivos `.dat`/`.hea` de PhysioNet, o EDF.
- Usar también `rr_prev` y `rr_next` en la decisión: hoy solo se mira la forma del latido, y una
  extrasístole supraventricular con QRS normal pasa por normal.

## Limitaciones que la página explica al médico

- Por construcción, cerca de 1 de cada 20 latidos normales queda por encima del umbral (el umbral
  es el percentil 95 de los N de val, D-010).
- La sensibilidad en val es del 56 %: muchos latidos anómalos no se marcan, y un resultado sin
  atípicos no descarta una arritmia.
- “Atípico” no es un diagnóstico. El modelo no distingue tipos de arritmia, ni evalúa ritmo,
  intervalos o el segmento ST.
- Solo se entrenó con MLII a 360 Hz y 18 registros de MIT-BIH.
