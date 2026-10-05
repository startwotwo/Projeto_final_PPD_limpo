#!/usr/bin/env python
"""Re-mede bh_gpu_full na grade do comparativo (apos fusao de kernels) e atualiza o JSON."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_gpu_vs_bh import run_one, N_PER_BACKEND, get_steps

P = Path(__file__).resolve().parents[2] / "benchmark_results" / "gpu_vs_bh_20261001_134501.json"
d = json.load(open(P))
d = [r for r in d if r.get("backend") != "bh_gpu_full"]
for N in N_PER_BACKEND["bh_gpu_full"]:
    steps = get_steps(N, 100)
    r = run_one("bh_gpu_full", N, steps)
    print(f"N={N}: {r['ms_per_step']:.2f} ms/step", flush=True)
    d.append(r)
    json.dump(d, open(P, "w"), indent=2)
print("JSON atualizado")
