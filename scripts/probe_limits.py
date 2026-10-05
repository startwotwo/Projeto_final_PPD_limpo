#!/usr/bin/env python
"""Sonda rapida de teto viavel por backend (1-3 steps, warmup fora do cronometro).

Uso: .\\.venv312\\Scripts\\python scripts/probe_limits.py
Saida incremental: benchmark_results/probe_limits.json (lista JSON reescrita a cada probe).
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import src.presets as presets
import src.presets_runner_fast as runner

DT = 0.005
EPS = 0.05
OUT = Path(__file__).resolve().parents[1] / "benchmark_results" / "probe_limits.json"

PROBES = [
    ("numba", 50_000, 3),
    ("numba", 100_000, 2),
    ("gpu", 200_000, 3),
    ("gpu", 500_000, 2),
    ("gpu_persist", 2_000_000, 2),
    ("bh", 50_000, 3),
    ("bh", 100_000, 2),
    ("bh_par", 100_000, 3),
    ("bh_par", 200_000, 2),
    ("bh_morton", 100_000, 3),
    ("bh_morton", 200_000, 2),
    ("mpi_manual", 50_000, 3),
    ("gpu_persist", 10_000_000, 1),  # teste da hipotese: N^2 em 10M e impraticavel?
]


def leapfrog(cf, pos, vel, masses, steps):
    pos_c = pos.copy()
    vel_c = vel.copy()
    accel = cf(pos_c, masses, 1.0, EPS)
    t0 = time.perf_counter()
    for _ in range(steps):
        vel_half = vel_c + accel * (DT * 0.5)
        pos_c = pos_c + vel_half * DT
        accel = cf(pos_c, masses, 1.0, EPS)
        vel_c = vel_half + accel * (DT * 0.5)
    return (time.perf_counter() - t0)


def main():
    results = []
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except Exception:
            results = []
    done = {(r["backend"], r["N"]) for r in results if r.get("success")}
    for backend, N, steps in PROBES:
        if (backend, N) in done:
            print(f"SKIP {backend} N={N} (ja medido)", flush=True)
            continue
        print(f"PROBE {backend} N={N} steps={steps}...", end=" ", flush=True)
        try:
            pos, vel, masses, _ = presets.generate_spiral_galaxy(
                N_disk=int(N * 0.7), N_bulge=N - int(N * 0.7), seed=42
            )
            if backend == "mpi_manual":
                import src.nbody_mpi_manual as mpi_man
                hosts = json.loads((Path(__file__).resolve().parents[1] / "hosts_manual.json").read_text(encoding="utf-8"))
                mpi_man.compute_forces_mpi_manual(pos, masses, G=1.0, eps=EPS, hosts=hosts)  # warmup
                pos_c = pos.copy(); vel_c = vel.copy()
                accel = mpi_man.compute_forces_mpi_manual(pos_c, masses, G=1.0, eps=EPS, hosts=hosts)
                t0 = time.perf_counter()
                for _ in range(steps):
                    vel_half = vel_c + accel * (DT * 0.5)
                    pos_c = pos_c + vel_half * DT
                    accel = mpi_man.compute_forces_mpi_manual(pos_c, masses, G=1.0, eps=EPS, hosts=hosts)
                    vel_c = vel_half + accel * (DT * 0.5)
                t_total = time.perf_counter() - t0
            else:
                cf, _, _ = runner._get_backends(backend)
                cf(pos, masses, 1.0, EPS)  # warmup (compilacao fora do cronometro)
                t_total = leapfrog(cf, pos, vel, masses, steps)
            r = {"backend": backend, "N": N, "steps": steps,
                 "total_time_sec": t_total, "ms_per_step": (t_total / steps) * 1000,
                 "success": True, "error": None}
            print(f"{r['ms_per_step']:.2f} ms/step ({t_total:.1f}s)", flush=True)
        except Exception as e:
            r = {"backend": backend, "N": N, "steps": steps,
                 "total_time_sec": None, "ms_per_step": None,
                 "success": False, "error": str(e)[:300]}
            print(f"FAILED: {str(e)[:120]}", flush=True)
        results.append(r)
        OUT.write_text(json.dumps(results, indent=2, default=str))
    print("Probes completos.")


if __name__ == "__main__":
    main()
