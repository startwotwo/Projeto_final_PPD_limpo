"""Benchmark sequencial — para tabela do relatório."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import time
import numpy as np
from src.nbody_sequential import SimConfig, run_simulation

for N in [500, 1000, 2000]:
    for tiled in [False, True] if N >= 1000 else [False]:
        cfg = SimConfig(N=N, steps=20, integrator="leapfrog", tiled=tiled, seed=42)
        # warmup para não medir import
        run_simulation(SimConfig(N=32, steps=2), verbose=False)
        t0 = time.perf_counter()
        result = run_simulation(cfg, verbose=False, energy_every=100)
        elapsed = time.perf_counter() - t0
        print(f"N={N:5d} tiled={str(tiled):5s}  elapsed={elapsed:.3f}s  per_step={elapsed/cfg.steps*1000:.1f}ms  Etot={result.energy_history[-1][2]:.2f}")

# Estimativa para N maiores (1 step só, para não esperar horas)
print("\n--- estimativa 1 step ---")
for N in [5000, 10000]:
    cfg = SimConfig(N=N, steps=1, integrator="leapfrog", tiled=True, seed=42)
    t0 = time.perf_counter()
    result = run_simulation(cfg, verbose=False, energy_every=100)
    elapsed = time.perf_counter() - t0
    print(f"N={N:5d} tiled=True  1 step={elapsed:.3f}s  estimado 100 steps={elapsed*100:.1f}s")
