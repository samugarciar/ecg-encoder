"""Pruebas unitarias para validar las particiones y constantes de configuración."""

from __future__ import annotations

import ecg.config as cfg


def test_splits_are_mutually_disjoint() -> None:
    """Verifica que no exista solapamiento entre train, val y test."""
    train_set = set(cfg.DS1_TRAIN)
    val_set = set(cfg.DS1_VAL)
    test_set = set(cfg.DS2_TEST)

    assert train_set.isdisjoint(val_set), "DS1_TRAIN y DS1_VAL comparten registros"
    assert train_set.isdisjoint(test_set), "DS1_TRAIN y DS2_TEST comparten registros"
    assert val_set.isdisjoint(test_set), "DS1_VAL y DS2_TEST comparten registros"


def test_total_records_count() -> None:
    """Verifica que la suma total de registros en los 3 splits sea exactamente 44."""
    all_splits = set(cfg.DS1_TRAIN) | set(cfg.DS1_VAL) | set(cfg.DS2_TEST)
    assert len(all_splits) == 44, f"Se esperaban 44 registros, pero hay {len(all_splits)}"
    # Validar longitudes individuales
    assert len(cfg.DS1_TRAIN) == 18
    assert len(cfg.DS1_VAL) == 4
    assert len(cfg.DS2_TEST) == 22


def test_no_excluded_records_in_splits() -> None:
    """Verifica que ninguno de los registros con marcapasos esté en los splits."""
    all_splits = set(cfg.DS1_TRAIN) | set(cfg.DS1_VAL) | set(cfg.DS2_TEST)
    excluded_set = set(cfg.EXCLUDED)

    assert all_splits.isdisjoint(excluded_set), (
        f"Registros de marcapasos presentes en los splits: {all_splits & excluded_set}"
    )


def test_window_parameters() -> None:
    """Verifica la coherencia de la ventana centrada asimétricamente en el pico R."""
    assert cfg.WIN_PRE == 90
    assert cfg.WIN_POST == 166
    assert cfg.WIN_LEN == 256
    assert cfg.WIN_PRE + cfg.WIN_POST == cfg.WIN_LEN


def test_aami_mapping_consistency() -> None:
    """Verifica que todas las asignaciones AAMI correspondan a las 5 clases estándar."""
    allowed_classes = {"N", "S", "V", "F", "Q"}
    for sym, cls in cfg.AAMI_MAP.items():
        assert cls in allowed_classes, f"Símbolo {sym} mapeado a clase inválida: {cls}"

    assert cfg.NORMAL_SYMBOLS == {"N"}
    assert cfg.AAMI_NORMAL == {"N", "L", "R", "e", "j"}
