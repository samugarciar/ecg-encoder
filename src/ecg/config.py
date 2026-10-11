"""Constantes globales del proyecto, configuración de señal y particiones inter-paciente."""

FS: int = 360
ADC_ZERO: int = 1024
ADC_GAIN: int = 200
LEAD: str = "MLII"
WIN_PRE: int = 90
WIN_POST: int = 166
WIN_LEN: int = WIN_PRE + WIN_POST  # 256 muestras [r-90, r+166)

BANDPASS: tuple[float, float] = (0.5, 40.0)
FILTER_ORDER: int = 3

EXCLUDED: list[int] = [102, 104, 107, 217]

DS1_VAL: list[int] = [116, 119, 209, 223]
DS1_TRAIN: list[int] = [
    101, 106, 108, 109, 112, 114, 115, 118, 122, 124,
    201, 203, 205, 207, 208, 215, 220, 230,
]
DS2_TEST: list[int] = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
    233, 234,
]

AAMI_MAP: dict[str, str] = {
    # Normal / Rama / Escape
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # Supraventricular
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # Ventricular
    "V": "V",
    "E": "V",
    # Fusión
    "F": "F",
    # Desconocido / No clasificable
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

NORMAL_SYMBOLS: set[str] = {"N"}  # Definición del proyecto: label=0 solo para N
AAMI_NORMAL: set[str] = {"N", "L", "R", "e", "j"}  # Vista AAMI para comparación externa

# Validaciones al importar
_train_set = set(DS1_TRAIN)
_val_set = set(DS1_VAL)
_test_set = set(DS2_TEST)
_excluded_set = set(EXCLUDED)

# 1. Los tres splits son mutuamente disjuntos
assert _train_set.isdisjoint(_val_set), "DS1_TRAIN y DS1_VAL no son disjuntos"
assert _train_set.isdisjoint(_test_set), "DS1_TRAIN y DS2_TEST no son disjuntos"
assert _val_set.isdisjoint(_test_set), "DS1_VAL y DS2_TEST no son disjuntos"

# 2. Su unión contiene exactamente 44 registros
_all_splits = _train_set | _val_set | _test_set
assert len(_all_splits) == 44, (
    f"Se esperaban 44 registros en los splits, pero se encontraron {len(_all_splits)}"
)

# 3. Ninguno contiene registros de EXCLUDED
assert _all_splits.isdisjoint(_excluded_set), "Los splits contienen registros marcados en EXCLUDED"
