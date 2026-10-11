# Detección de anomalías en ECG con autoencoder

## Contexto del proyecto
- Proyecto final de la asignatura de Redes Neuronales Artificiales (RNA), Ingeniería de Sistemas e Informática, Universidad Nacional de Colombia, Sede Medellín. El mismo proyecto se usa también para el seminario de data science e IA.
- Plazo: 2 a 4 semanas.
- Equipo: un integrante (Andrés) tiene experiencia previa en ML y competitive programming. El resto del equipo no tiene experiencia previa en deep learning. Nadie en el equipo tiene experiencia previa en bioingeniería ni en procesamiento de señales.
- Framework: se permite usar librerías estándar (PyTorch/TensorFlow/Keras), no se exige implementar desde cero.

## El problema, en una frase
Entrenar un autoencoder para reconstruir latidos de ECG normales. Un latido que el modelo reconstruye mal (error de reconstrucción alto) se marca como posible anomalía (arritmia).

## Por qué este enfoque y no clasificación supervisada
- Las arritmias son raras frente a los latidos normales (desbalance de clases severo). Un clasificador supervisado tiende a aprender solo las anomalías que vio en entrenamiento y generaliza mal a tipos de arritmia poco representados.
- El autoencoder solo necesita latidos normales para entrenar. Cualquier cosa que se reconstruya mal es candidata a anómala, incluidas arritmias que el modelo nunca vio etiquetadas.
- Trade-off: clasificación es "cerrada" (detecta lo que etiquetaste), reconstrucción es "abierta" (detecta desviaciones de lo normal). Para arritmias, donde lo raro es raro por definición, eso importa más que la arquitectura en sí.

## Marco teórico que hay que entender antes de programar
1. **Autoencoder**: red con un encoder que comprime la entrada a una representación de menor dimensión (cuello de botella) y un decoder que intenta reconstruir la entrada original. Se entrena minimizando el error de reconstrucción (MSE típicamente), sin etiquetas.
2. **Elección de capa (Conv1D, no LSTM/RNN)**: si se trabaja con ventanas cortas centradas en cada latido (segmentos alrededor del pico R), no hay dependencia de largo alcance que justifique recurrencia. Conv1D entrena más rápido y más estable. LSTM se justifica si se trabaja con tiras largas multi-latido en vez de latidos individuales.
3. **Umbral de anomalía**: no hay un punto de corte automático. Lo típico es tomar un percentil del error de reconstrucción sobre un set de validación solo con normales (ej. percentil 95). Es una decisión con trade-off explícito entre falsos positivos y falsos negativos, hay que justificarla en el informe, no es un detalle técnico que se resuelve solo.
4. **Métricas para clases desbalanceadas**: no usar solo accuracy (con pocas anomalías, un modelo que siempre dice "normal" ya tiene accuracy alta y es inútil). Usar precision, recall, F1 y AUC-ROC sobre la clase anómala.

## Datasets
- **ECG5000**: 5000 latidos ya segmentados, listos para usar, disponible en el UCR Time Series Archive. Ventaja: no requiere procesar archivos crudos ni detectar picos R, se puede tener un pipeline funcionando en días. Limitación importante: proviene de una sola grabación de 20 horas de un único paciente (registro chf07, base BIDMC-CHFDB), con etiquetas generadas por un detector automático, nunca corregidas manualmente por un cardiólogo. No prueba que el modelo generalice entre personas.
- **MIT-BIH Arrhythmia Database** (PhysioNet): grabaciones de 48 pacientes distintos, etiquetadas por cardiólogos. Requiere la librería `wfdb` (paquete oficial de PhysioNet) para leer los archivos `.dat` (señal) y `.atr` (anotaciones de cada latido).
- **Plan sugerido**: usar ECG5000 la primera semana para tener el pipeline completo funcionando rápido (carga de datos, arquitectura, entrenamiento, umbral, métricas). Si el tiempo alcanza, migrar a MIT-BIH crudo para el resultado final, porque es multi-paciente y es lo que de verdad valida que el enfoque generaliza.

## Riesgos metodológicos que hay que evitar
- **Split por paciente, no por latido**: si se separan train/test por latido individual en vez de por paciente, latidos del mismo paciente terminan en ambos lados. Como la morfología del ECG es bastante paciente-específica, esto infla el desempeño de forma artificial. Es el error metodológico más común en trabajos de ML con ECG.
- **No reportar solo accuracy** (ver punto 4 del marco teórico).
- **No mezclar el resto de la asignatura de RNA con el seminario sin verificar**: ambas asignaturas comparten el mismo proyecto, pero puede que difieran en el entregable esperado (ver preguntas abiertas).

## Bibliografía y recursos
**Fundamentos de deep learning**
- 3Blue1Brown, serie "Neural Networks" (YouTube). Intuición visual, sin código.
- StatQuest (Josh Starmer), videos de redes neuronales y autoencoders.
- fast.ai, curso "Practical Deep Learning for Coders" (gratis, con notebooks).

**Autoencoders**
- François Chollet, "Building Autoencoders in Keras" (blog.keras.io). Referencia canónica, cubre autoencoder simple, profundo, convolucional y secuencia a secuencia.
- DigitalSreeni (YouTube): "Applications of Autoencoders - Anomaly Detection" y "LSTM Autoencoder for anomaly detection".

**El mismo problema ya resuelto por otros (para comparar arquitectura y resultados)**
- Arrhythmia classification of LSTM autoencoder based on time series anomaly detection (ScienceDirect, 2021). Autoencoder LSTM + preprocesamiento de ECG sobre MIT-BIH, prácticamente el mismo proyecto.
- Abnormal ECG detection based on an adversarial autoencoder, "ECG-AAE" (Frontiers in Physiology, 2022). Más ambicioso, útil para ver el techo de lo que da este enfoque.
- A convolutional autoencoder framework for ECG signal analysis (ScienceDirect, 2025). Referencia si Conv1D resulta más simple que LSTM.

**Bioingeniería y señal desde cero**
- Serie de artículos "Working with MIT-BIH ECG Data in Python" (Proto Bioengineering, Medium). Explica cómo abrir los archivos crudos de MIT-BIH con WFDB, incluidas las anotaciones de cardiólogos.
- `wfdb-python` (repositorio oficial de PhysioNet en GitHub), con notebook de ejemplo.

**Implementación de referencia**
- Venelin Valkov, "Time Series Anomaly Detection Tutorial with PyTorch in Python | LSTM Autoencoder for ECG Data" (YouTube), con notebook del libro *Getting Things Done with PyTorch*. Código completo y funcional para exactamente este problema.

## Roles sugeridos (a ajustar según el tamaño real del equipo)
- Carga y preprocesamiento de datos (parseo de ECG5000 y/o MIT-BIH con wfdb, segmentación de latidos, normalización).
- Arquitectura y entrenamiento del autoencoder.
- Selección de umbral y evaluación (precision, recall, F1, AUC-ROC, curvas de error de reconstrucción).
- Informe y comparación de resultados frente a los papers de referencia listados arriba.

## Preguntas abiertas (Andrés todavía no las ha definido, hay que resolverlas antes de repartir trabajo en serio)
- Cuántas personas hay exactamente en el equipo.
- Qué evalúa el seminario de forma específica: informe tipo mini-paper, demo funcionando, comparación de métodos.
- Acceso real a GPU del equipo, más allá del computador de Andrés.
