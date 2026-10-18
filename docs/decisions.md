# Registro de Decisiones de Arquitectura y Metodología (ADR)

Este documento registra las decisiones metodológicas y técnicas tomadas en el proyecto, su justificación y las alternativas evaluadas.

---

### D-001: MIT-BIH en CSV en vez de ECG5000
- **Decisión**: Utilizar la base de datos MIT-BIH Arrhythmia Database (en formato CSV con anotaciones) como fuente principal.
- **Por qué**: Es un dataset multi-paciente (48 grabaciones de 30 minutos), anotado manualmente por cardiólogos y con los picos R ya localizados y validados clínicamente. Permite evaluar la generalización real entre sujetos.
- **Alternativa descartada**: ECG5000 (UCR Archive), descartado porque proviene de un único paciente (grabación chf07 de BIDMC-CHFDB) y fue etiquetado automáticamente sin validación clínica manual, lo que imposibilita validar generalización inter-paciente.

---

### D-002: Excluir registros con marcapasos (102, 104, 107, 217)
- **Decisión**: Excluir los registros 102, 104, 107 y 217 del conjunto de estudio.
- **Por qué**: Estándar AAMI EC57. Estos registros corresponden a pacientes con marcapasos artificiales, cuyas espigas de estimulación introducen artefactos morfológicos atípicos ajenos a la fisiología intrínseca del corazón.
- **Alternativa descartada**: Incluir todos los 48 registros, lo cual distorsionaría la reconstrucción morfológica estándar.

---

### D-003: Split inter-paciente DS1/DS2 (de Chazal et al., 2004)
- **Decisión**: Dividir los 44 registros válidos siguiendo la partición inter-paciente canónica de Chazal (DS1 y DS2). Para validación interna de DS1 se asignan los registros `[116, 119, 209, 223]`, dejando 18 registros en entrenamiento y 22 en test (DS2). Nota: 201 (train) y 202 (test) corresponden al mismo paciente; se mantiene la partición original para comparar con la literatura (ver D-013).
- **Por qué**: Evitar el error metodológico más común en ECG: dividir por latidos (intra-paciente). La morfología cardíaca es paciente-específica; evaluar en pacientes nunca vistos es la única prueba real de generalización.
- **Alternativa descartada**: Validación cruzada aleatoria por latidos (infla métricas de forma artificial por fuga de información de morfología del mismo sujeto).

---

### D-004: Utilización exclusiva de la derivación MLII
- **Decisión**: Utilizar únicamente la derivación de derivación modificada II (MLII) para la extracción y reconstrucción de latidos.
- **Por qué**: Es la única derivación presente de manera homogénea y continua a lo largo de los 44 registros seleccionados de la base de datos.
- **Alternativa descartada**: Utilizar derivaciones mixtas (V1, V2, V4, V5), lo que generaría inconsistencias en canales ausentes en varios registros.

---

### D-005: Definición de normalidad (Normal = solo 'N') y reporte de vista AAMI
- **Decisión**: Para el objetivo del autoencoder, se etiquetan como normales (`label = 0`) estrictamente los latidos con símbolo `'N'`. Latidos como `L` (bloqueo de rama izquierda), `R` (bloqueo de rama derecha), `e` y `j` se tratan como anomalías (`label = 1`). En la evaluación se incluye adicionalmente la `aami_view` (donde `N, L, R, e, j` se consideran normales) para contrastar con la literatura.
- **Por qué**: Desde la perspectiva no supervisada de reconstrucción, una alteración en la conducción intraventricular (`L`, `R`) altera la morfología de activación ventricular respecto al latido sinusal típico `N`. Ofrecer ambas métricas permite rigor interno y comparabilidad externa.
- **Alternativa descartada**: Mezclar `L` y `R` en el entrenamiento de normales, lo que debilitaría la capacidad del modelo de especializarse en la forma sinusal estricta.

---

### D-006: Ventana de 256 muestras centrada asimétricamente en el pico R [r-90, r+166)
- **Decisión**: Extraer segmentos de 256 muestras tomando 90 muestras previas al pico R y 166 muestras posteriores a 360 Hz (~711 ms).
- **Por qué**: Esta ventana cubre holgadamente desde la onda P hasta el final de la onda T para ritmos típicos a taquicárdicos, y 256 es potencia de 2, óptima para convoluciones y reducciones sucesivas de dimensionalidad en Conv1D.
- **Alternativa descartada**: Ventanas simétricas (ej. 128 antes y 128 después) que cortan la onda T prematuramente o ventanas no potencias de 2 que complican el padding en arquitecturas convolucionales.

---

### D-007: Filtrado Butterworth orden 3 (0.5–40 Hz) con fase cero sobre el registro continuo
- **Decisión**: Aplicar filtro pasa banda Butterworth de orden 3 entre 0.5 Hz y 40.0 Hz usando filtrado bidireccional de fase cero (`sosfiltfilt`) directamente sobre la señal continua completa del registro antes de segmentar.
- **Por qué**: Elimina la deriva de la línea base (respiración, <0.5 Hz) y la interferencia de la red eléctrica / ruido muscular (>40 Hz) sin introducir distorsiones de fase ni artefactos de borde en los latidos individuales.
- **Alternativa descartada**: Filtrar latidos ya recortados (genera artefactos transitorios severos en los extremos de cada latido).

---

### D-008: Normalización z-score por latido individual
- **Decisión**: Aplicar estandarización z-score `(x - mean) / std` de forma independiente a cada segmento de latido recortado.
- **Por qué**: Neutraliza variaciones de impedancia de contacto del electrodo y escala de amplitud entre pacientes, forzando al modelo a aprender características puramente morfológicas. Queda pendiente el experimento contrastando con señal sin normalizar.
- **Alternativa descartada**: Escalado Min-Max global, susceptible a picos de artefactos transitorios que comprimen el rango dinámico del QRS.

---

### D-009: Arquitectura base Autoencoder Conv1D con cuello de botella latente de dimensión 32
- **Decisión**: Utilizar una red neuronal autoencoder completamente convolucional 1D con compresión a dimensión 32 como baseline.
- **Por qué**: Las convoluciones 1D capturan correlaciones locales de forma más eficiente y con órdenes de magnitud menor tiempo de entrenamiento que arquitecturas recurrentes (LSTM/GRU). Para latidos individuales segmentados de longitud fija, no se requiere recurrencia de largo alcance.
- **Alternativa descartada**: LSTM Autoencoder (más lento de entrenar, mayor varianza en convergencia y sin ventaja empírica en segmentos cortos).

---

### D-010: Selección de umbral de anomalía sobre el percentil 95 en validación
- **Decisión**: Determinar el umbral de anomalía mediante el percentil 95 del error de reconstrucción obtenido en los latidos normales del conjunto de validación (`DS1_VAL`). El split de test (`DS2_TEST`) nunca interviene en la selección del umbral.
- **Por qué**: Fija una tasa de falsos positivos controlada a priori (~5%) sobre datos no vistos en entrenamiento pero pertenecientes al split de desarrollo, respetando el aislamiento estricto de test.
- **Alternativa descartada**: Optimizar el umbral buscando maximizar F1 en test (constituye data snooping / fuga metodológica inaceptable).

---

### D-011: Framework de Deep Learning en PyTorch
- **Decisión**: Desarrollar los modelos y el bucle de entrenamiento en PyTorch (2.5+).
- **Por qué**: Control explícito del flujo computacional, reproducibilidad mediante fijación de semillas deterministas y amplia adopción académica.
- **Alternativa descartada**: TensorFlow/Keras o implementaciones manuales desde cero.

---

### D-012: Evaluación única sobre DS2 con modelo congelado
- **Decisión**: El conjunto `DS2_TEST` permanece intocado durante todo el proceso de ingeniería, ajuste de hiperparámetros y selección de umbrales. Solo se ejecuta una vez al final por la persona designada como guardiana del test, con el modelo y pesos congelados (`models/frozen/`).
- **Por qué**: Blindar el experimento contra sobreajuste al conjunto de prueba y garantizar la validez científica del reporte final.
- **Alternativa descartada**: Evaluar repetidamente sobre test para tomar decisiones de diseño o detener el entrenamiento.

---

### D-013: Mantener la partición de de Chazal con 201 en train y 202 en test
- **Decisión** (13 oct 2026): Se mantiene la partición de de Chazal et al. (2004) tal cual: el registro 201 queda en `DS1_TRAIN` y el 202 en `DS2_TEST`, aunque ambos provienen del mismo paciente.
- **Por qué**: Permite comparar directamente con la literatura que usa la partición DS1/DS2. Mover o excluir uno de los dos registros rompería esa comparabilidad.
- **Limitación**: Queda una posible fuga de morfología paciente-específica entre train y test a través de este par. Se documenta como limitación del estudio, y se sugiere a Evaluación reportar las métricas de DS2 con y sin el registro 202.
- **Alternativa descartada**: Pasar 201 a test o 202 a train (o excluir uno de los dos). Elimina la fuga, pero los resultados dejan de ser comparables con los trabajos que usan la partición estándar.

---

### D-014: Solo los 15 símbolos de `AAMI_MAP` son latidos
- **Decisión** (16 oct 2026): Entran a C1 únicamente las anotaciones con uno de los 15 símbolos de `config.AAMI_MAP` (`N L R e j A a J S V E F / f Q`). Quedan fuera las 3 153 anotaciones que no son latidos: `+` (1 291, cambios de ritmo), `~` (616, calidad de señal), `!` (472, ondas de aleteo ventricular, todas en el 207), `"` (437, comentarios), `x` (193, ondas P no conducidas), `|` (132, artefactos), `[` y `]` (6 y 6, inicio y fin de fibrilación ventricular).
- **Por qué**: Es la práctica estándar con AAMI EC57 y permite comparar los conteos con la literatura. Ninguna de las anotaciones excluidas es un complejo QRS clasificable.
- **Alternativa descartada**: Contar las ondas `!` del 207 como latidos ventriculares. Son ondas de aleteo, no QRS individuales, y no tienen clase AAMI.

---

### D-015: `config.py` es la única fuente de parámetros de datos y splits
- **Decisión** (16 oct 2026): `src/ecg/config.py` es la única fuente de los parámetros de señal (fs, conversor A/D, derivación), de ventana, de filtro y de los splits. Se elimina `configs/data.yaml`, que repetía esos valores y no lo leía ningún módulo. `configs/` queda para los hiperparámetros de entrenamiento (regla 8).
- **Por qué**: Dos copias terminan desincronizándose. La regla 5 reserva los splits para `config.py`, que además los valida al importar. La ventana y el filtro no son hiperparámetros que se ajusten: forman parte del contrato C1 y quedan registrados en `meta.json`.
- **Alternativas descartadas**: Que `config.py` lea los parámetros desde el YAML (cambia un módulo compartido sin necesidad), o quitar solo los splits del YAML (deja fs, ventana y filtro duplicados).

---

### D-016: `mlii_noisy` se marca si cualquier muestra de la ventana cae en un tramo ruidoso
- **Decisión** (16 oct 2026): `mlii_noisy = True` si alguna muestra de la ventana `[r-90, r+166)` cae en un tramo con MLII marcada como ruidosa. Los tramos se leen de las marcas `~` como un interruptor: cada marca fija el estado hasta la siguiente. El subtipo es un campo de bits (bit 0 = señal 0, bit 1 = señal 1); MLII es la señal 0 en todos los registros salvo el 114, donde es la 1. El subtipo `-1` (8 casos) tiene todos los bits encendidos y cuenta como ruidoso.
- **Por qué**: El autoencoder recibe la ventana completa, no solo el pico R. Un latido cuya ventana toca un tramo ruidoso tiene la entrada contaminada, así que este es el criterio más conservador para las métricas `clean_only`.
- **Alternativa descartada**: Mirar solo la muestra del R. Es más simple, pero deja como limpios los latidos cuyo inicio o final cae en un tramo ruidoso. Antes de descartar bordes, el criterio de ventana marca 908 / 46 / 1 076 latidos (train/val/test) y el del R, 852 / 41 / 1 002.

---

### D-017: El RR se calcula entre latidos AAMI consecutivos, antes de descartar bordes
- **Decisión** (16 oct 2026): `rr_prev` y `rr_next` se calculan registro por registro, entre latidos consecutivos con símbolo en `AAMI_MAP`, **antes** de descartar los 39 latidos cuya ventana no cabe en el registro. Solo el primer latido de cada registro tiene `rr_prev = NaN` y solo el último `rr_next = NaN`; si ese latido es de borde, se descarta junto con su `NaN`.
- **Por qué**: El RR es una propiedad fisiológica: los latidos de borde existieron aunque su ventana no quepa. Calcularlo después cambiaría el RR de 39 latidos, los vecinos de los bordes. En esta base todas las anotaciones de latido son los 15 símbolos AAMI, así que "solo latidos AAMI" y "todos los QRS" coinciden.
- **Limitación**: Los RR que cruzan episodios sin latidos anotados salen largos. En DS1 hay 22 latidos con `rr_prev` mayor de 2 s (registros 201, 207 y 208); el más largo, de 100 s, cruza el episodio de aleteo ventricular del 207 (minutos 25,7 a 27,3), marcado solo con ondas `!`.
- **Alternativa descartada**: Calcular el RR después de quitar los bordes.

---

### D-018: "Normal" es el símbolo `N` en cualquier ritmo
- **Decisión** (16 oct 2026): `label = 0` sigue siendo el símbolo `N`, sin mirar el ritmo anotado (D-005). C1 no guarda el ritmo.
- **Por qué**: Mantiene el contrato y la comparabilidad con la literatura, que define las clases por símbolo.
- **Limitación**: 4 336 de los 29 607 latidos `N` de train (14,6 %) ocurren en fibrilación auricular (201 y 203), flutter auricular (203) o preexcitación (230). El autoencoder aprende también esas formas como normales, lo que puede bajar su sensibilidad a latidos supraventriculares. Se reporta en el informe.
- **Alternativas descartadas por ahora**: Agregar una columna `rhythm` a C1 para que Modelo pruebe entrenar solo con `N` sinusales (cambio de contrato aditivo; se puede retomar con Modelo), o redefinir "normal" como `N` en ritmo sinusal (rompe D-005 y la comparación con la literatura).

---

### D-019: El pipeline de datos se documenta en un notebook y se implementa en `src/ecg/data/`
- **Decisión** (16 oct 2026): `notebooks/datos_mitbih_c1.ipynb` queda como el documento por etapas del pipeline (explicaciones, figuras y comprobaciones). El código definitivo pasa a `src/ecg/data/` (`records.py`, `filtering.py`, `segmentation.py`, `build.py`) con tests, y C1 se genera con `python -m ecg.data.build`. Cuando esos módulos existan, el notebook los importa en vez de definir sus propias funciones. Mientras tanto, el notebook genera C1.
- **Por qué**: El notebook sirve para explicar y para mirar los datos; los módulos se prueban en CI (regla 10) y dan al equipo un comando reproducible.
- **Alternativa descartada**: Dejar el notebook como generador oficial de C1, sin tests en CI ni un comando para el resto del equipo.
