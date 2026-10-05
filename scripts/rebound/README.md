# scripts/rebound — backend alternativo REBOUND/IAS15

Experimentos com o integrador IAS15 (REBOUND) como referência de alta precisão
para a validação lunar (§10.6 do README). Requer `pip install rebound`.

```powershell
.\.venv312\Scripts\python scripts/rebound/record_rebound.py --years 10 --dt 0.01 --out data/rebound_430.npz
.\.venv312\Scripts\python scripts/rebound/play_rebound.py --file data/rebound_430.npz --follow Jupiter
```

- `record_rebound.py` — propaga `solar_completo` com IAS15 e grava a trajetória em NPZ
- `play_rebound.py` — visualiza um NPZ gravado (`--follow`, `--trails`, `--glow`)
- `live_vispy_rebound.py` — IAS15 ao vivo com follow
- `sim_rebound.py`, `sim_rebound_vectors.py` — conversores Horizons→REBOUND
  (ELEMENTS e VECTORS; o modo VECTORS funciona para todos os corpos)
