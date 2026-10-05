# scripts/archive — experimentos superados e ferramentas pontuais

Arquivos movidos para cá por não fazerem parte do fluxo principal (ver `scripts/README.md`
se existir, senão a raiz `README.md`). Rodam de onde estão, chamados da raiz do repo:

```powershell
.\.venv312\Scripts\python scripts/archive/<arquivo>.py [args]
```

Os caminhos internos usam `parents[2]` (= raiz do projeto), então funcionam no novo local.

- **Variantes live superadas**: `live_demo.py`, `live_6_anim.py`, `live_maxfps.py`,
  `live_vispy_6.py`, `live_vispy_bloom.py`, `live_vispy_bloom_fix.py`
  (o visualizador oficial é `scripts/live_vispy.py`)
- **Benchmarks antigos**: `benchmark_backends.py`, `benchmark_sequential.py`,
  `compara_6.py`, `demo_desempenho.py` (a suíte oficial é `run_benchmarks.py`)
- **Ferramentas pontuais** (usadas uma vez, guardadas por reprodutibilidade):
  `diag_jwst.py`, `diag_jwst_prop.py`, `diag_prefix.py`,
  `rerun_bh_full.py`, `rerun_bh_full_main.py`, `rerun_bh_full_wall.py`,
  `run_one_fresh.py`
- **Outros**: `build_sim_func.py` (superseded por `src/presets.py`),
  `patch_nb.py` (patch pontual de notebook), `fetch_saturn_all.py` (placeholder),
  `hosts_local.json`, `hosts_local2.json` (configs locais não usadas)
