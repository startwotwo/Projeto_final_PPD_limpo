import json, pathlib
nb=json.loads(pathlib.Path('notebooks/Projeto_NBody_Completo.ipynb').read_text(encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    if 'BH vs direto' in src:
        idx=i
        break
else:
    idx=None
print('found',idx)
md = {'cell_type':'markdown','metadata':{},'source':['### §6.2 — Novos modos com build paralelo (demonstração)\n','Os modos `bh_mp` (build por octante em `ThreadPool 8t`) e `bh_morton` (geração Morton `prange 28t` + sort) mostram que o gargalo do BH é o `build` sequencial por frame. `bh_mp`/`bh_morton` mantêm `traversal` paralelo. Para `N=5k` `bh_mp` já supera `bh` serial; `gpu N²` ainda vence até `~30k` (ver §7). Código em `src/nbody_bh_parallel_build.py:1` e `src/presets_runner_fast.py:1`.\n']}
code_src = """# §6.2 — Comparativo BH novos modos (build paralelo demo)
try:
    from src.nbody_barnes_hut import compute_forces_bh
    from src.nbody_bh_parallel_build import compute_forces_bh_mp, compute_forces_bh_morton
    from src.nbody_sequential import compute_forces_tiled
    from src.presets import generate_plummer_sphere
except ImportError:
    from nbody_barnes_hut import compute_forces_bh
    from nbody_bh_parallel_build import compute_forces_bh_mp, compute_forces_bh_morton
    from nbody_sequential import compute_forces_tiled
    from presets import generate_plummer_sphere
import time, numpy as np
for N in [2000,5000,10000]:
    pos,vel,masses=generate_plummer_sphere(N,seed=42)
    print(f"\\nN={N}")
    for name,fn in [("bh",compute_forces_bh),("bh_mp",compute_forces_bh_mp),("bh_morton",compute_forces_bh_morton),("numba_tiled",lambda p,m: compute_forces_tiled(p,m,eps=0.05))]:
        t0=time.perf_counter(); a=fn(pos,masses,eps=0.05); t=(time.perf_counter()-t0)*1000
        print(f" {name:12s} {t:7.1f} ms")
print("\\nObs: build por frame é sequencial em bh; bh_mp paraleliza por octante (8 tasks), bh_morton paraleliza Morton (prange). Ganho cresce com N e uniformidade.")
"""
code = {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source': code_src.splitlines(True)}
nb['cells'].insert(idx+2, md)
nb['cells'].insert(idx+3, code)
pathlib.Path('notebooks/Projeto_NBody_Completo.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print('inserted', len(nb['cells']))
