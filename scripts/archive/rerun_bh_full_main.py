#!/usr/bin/env python
"""Re-mede bh_gpu_full na grade da suite principal (apos fusao) e atualiza o JSON."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_benchmarks as RB

P = Path(__file__).resolve().parents[2] / "benchmark_results" / "benchmark_20261001_132909.json"
d = json.load(open(P))
d = [r for r in d if r.get("backend") != "bh_gpu_full"]
for N in RB.N_PER_BACKEND["bh_gpu_full"]:
    steps = RB.get_steps("bh_gpu_full", N)
    r = RB.run_benchmark("bh_gpu_full", N, steps)
    print(f"N={N}: {r['ms_per_step']:.2f} ms/step", flush=True)
    d.append(r)
    json.dump(d, open(P, "w"), indent=2)
print("JSON atualizado")
