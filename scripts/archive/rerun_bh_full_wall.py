#!/usr/bin/env python
"""Re-mede bh_gpu_full na grade do run 134332 (500->10M) com o codigo atual.

Mantem o ponto de 20M original (parede de VRAM/swap — nao reexecutado por
decisao operacional) com nota explicita. Atualiza o JSON no lugar.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_gpu_vs_bh import run_one, get_steps

P = Path(__file__).resolve().parents[2] / "benchmark_results" / "gpu_vs_bh_20260908_134332.json"
d = json.load(open(P))
keep = []
for r in d:
    if r.get("backend") == "bh_gpu_full" and r.get("N") == 20000000 and r.get("success"):
        r = dict(r)
        r["note"] = ("codigo pre-fusao (679ms em 10M na epoca); ponto mantido como "
                     "demonstracao do muro de VRAM/swap em 20M, nao reexecutado")
        keep.append(r)
grid = [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 350000,
        500000, 1000000, 2000000, 5000000, 10000000]
for N in grid:
    steps = get_steps(N, 100)
    r = run_one("bh_gpu_full", N, steps)
    print(f"N={N}: {r['ms_per_step']:.2f} ms/step", flush=True)
    keep.append(r)
    json.dump(keep, open(P, "w"), indent=2)
print("JSON 134332 atualizado (500-10M codigo novo + 20M original anotado)")
