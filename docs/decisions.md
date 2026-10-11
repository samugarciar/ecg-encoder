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
- **Decisión**: Dividir los 44 registros válidos siguiendo la partición inter-paciente canónica de Chazal (DS1 y DS2). Para validación interna de DS1 se asignan los registros `[116, 119, 209, 223]`, dejando 18 registros en entrenamiento y 22 en test (DS2). Nota: 201 y 202 corresponden al mismo sujeto, ambos ubicados en el mismo split para evitar fuga.
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
