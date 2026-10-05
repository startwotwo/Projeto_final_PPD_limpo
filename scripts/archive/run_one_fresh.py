#!/usr/bin/env python
"""Mede UM ponto bh_gpu_full em processo fresco (pool limpo, sem historico).

Uso: python scripts/archive/run_one_fresh.py N [trials] [json_out]
Anexa {"backend","N","ms_per_step",...} ao JSON de saida.
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import src.presets as presets
from src.nbody_bh_gpu_build import compute_forces_bh_gpu_device as full

N = int(sys.argv[1])
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = sys.argv[3] if len(sys.argv) > 3 else None

pos, vel, masses, _ = presets.generate_spiral_galaxy(
    N_disk=int(N * 0.7), N_bulge=N - int(N * 0.7), seed=42)
full(pos, masses, 1.0, 0.05)  # warmup: compilacao + pools fora do cronometro
vals = []
for t in range(TRIALS):
    a = pos.copy()
    t0 = time.perf_counter()
    full(a, masses, 1.0, 0.05)
    vals.append((time.perf_counter() - t0) * 1000)
ms = float(np.median(vals))
print(f"FRESH N={N}: {ms:.2f} ms/step trials={[round(v, 1) for v in vals]}", flush=True)
if OUT:
    p = Path(OUT)
    d = json.load(open(p)) if p.exists() else []
    d = [r for r in d if not (r.get("backend") == "bh_gpu_full" and r.get("N") == N)]
    d.append({"backend": "bh_gpu_full", "N": N, "steps": 1,
              "total_time_sec": ms / 1000.0, "ms_per_step": ms,
              "success": True, "error": None,
              "note": f"processo fresco dedicado, mediana de {TRIALS} trials"})
    json.dump(d, open(p, "w"), indent=2)
