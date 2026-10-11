## Qué cambia
<!-- Describe de forma concisa qué modificaciones o adiciones introduce este PR -->

## Cómo verificarlo
<!-- Pasos concretos para probar y verificar los cambios introducidos -->

## Resultados (JSON o figuras)
<!-- Incluye fragmentos de salida, métricas JSON relevantes o referencias a figuras generadas -->

## ¿Toca un contrato?
<!-- Indicar si este PR altera docs/contracts.md (C1, C2, C3). Si es así, etiquetar como 'contract-change' -->
- [ ] No modifica contratos existentes
- [ ] Modifica contratos y actualiza `docs/contracts.md` correspondientemente

## Checklist
- [ ] `pytest -q` pasa en verde
- [ ] `ruff check .` pasa en verde
- [ ] Sin datos ni pesos en el commit (`data/raw/`, `data/processed/`, `runs/`, `*.npz`, `*.pt`)
- [ ] No utiliza el split "test" (DS2) en desarrollo
