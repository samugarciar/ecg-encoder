# Datos de ECG: MIT-BIH Arrhythmia Database

## Regla de Oro
**`data/raw/` es de estricta solo lectura.** Nunca modifiques, renombres, agregues ni elimines archivos dentro de este directorio. Todo dato procesado se almacena exclusivamente en `data/processed/`.

## Estructura
```
data/
├── raw/
│   └── mitbih_database/   # 96 archivos (48 registros .csv y 48 annotations.txt)
├── processed/             # beats.npz, meta.json generados por el pipeline
├── MANIFEST.sha256        # Sumas de verificación SHA-256 de los 96 archivos crudos
└── README.md
```

## Verificación de Integridad
Para comprobar que los datos en `data/raw/mitbih_database/` no han sido alterados ni corrompidos, ejecuta:
```bash
python scripts/verify_raw_data.py
```
El script reportará `OK` si los 96 archivos coinciden exactamente con sus hashes SHA-256 en `data/MANIFEST.sha256`.
