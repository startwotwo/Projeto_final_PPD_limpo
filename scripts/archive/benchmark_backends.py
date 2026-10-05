"""Compara sequencial vs numba — para relatório speedup."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.presets import generate_spiral_galaxy
from src.nbody_sequential import compute_forces_tiled
import src.nbody_numba as nb
import numpy as np

print(f"Numba threads: {nb.info()['threads']}  layer={nb.info()['threading_layer']}")
print(f"CPU cores lógico 28, físico 20")

for N in [500, 1000, 2000, 5000]:
    Nd = int(N*0.7); Nb_= N-Nd
    pos, vel, masses, _ = generate_spiral_galaxy(N_disk=Nd, N_bulge=Nb_, seed=42)
    # warmup
    nb.compute_forces_numba(pos, masses, 1.0, 0.05)
    for _ in range(3):
        compute_forces_tiled(pos, masses, 1.0, 0.05)

    # seq
    t0=time.perf_counter()
    r=5
    for _ in range(r):
        compute_forces_tiled(pos, masses, 1.0, 0.05)
    t_seq=(time.perf_counter()-t0)/r*1000
    # numba
    t0=time.perf_counter()
    for _ in range(20):
        nb.compute_forces_numba(pos, masses, 1.0, 0.05)
    t_nb=(time.perf_counter()-t0)/20*1000
    speedup=t_seq/max(t_nb,1e-9)
    print(f"N={N:4d}  seq={t_seq:7.2f} ms  numba={t_nb:6.2f} ms  speedup={speedup:5.1f}x  eff={speedup/28*100:4.1f}%")

# teste simulação longa
print("\n--- Simulação longa (colisão) ---")
from src.presets import generate_galaxy_collision
from src.presets_runner_fast import run_fast
from src.presets_runner import run_custom_simulation
import time as tm

pos, vel, masses, labels = generate_galaxy_collision(N_disk1=400, N_bulge1=150, N_disk2=400, N_bulge2=150, separation=6, impact_param=1.0, rel_vel_factor=0.45, seed=42)
for backend, func in [("seq", lambda: run_custom_simulation(pos, vel, masses, steps=300, dt=0.005, save_trajectory=False, verbose=False)),
                      ("numba", lambda: run_fast(pos, vel, masses, steps=300, dt=0.005, save_trajectory=False, verbose=False, backend="numba"))]:
    t0=tm.perf_counter()
    res=func()
    elapsed=tm.perf_counter()-t0
    print(f"{backend:5s} 300 steps N=1100  {elapsed:.2f}s  {elapsed/300*1000:.2f} ms/step")
