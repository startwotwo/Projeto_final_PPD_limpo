# scripts/validation — validação lunar Jan→Set/2026 (§10.6 do README)

Scripts que sustentam a comparação simulação vs. efemérides reais:

- `check_moon_phase.py [cache]` — fase/elongação da Lua direto do cache Horizons
- `moon_curve.py` — curva de fase ao longo do ano
- `moon_err_decomp.py` — decomposição do erro (integrador vs. modelo)
- `moon_ias15.py`, `moon_ias15_full.py` — referência de precisão com IAS15

Chamados da raiz do repo, ex.:
```powershell
.\.venv312\Scripts\python scripts/validation/check_moon_phase.py data/horizons_cache_20260928.json
```
