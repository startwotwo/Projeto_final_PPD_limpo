"""
Compara Euler vs Leapfrog por preset — drift de energia vs passos
Uso:
  .\.venv\Scripts\python scripts/compare_integrators.py --preset galaxy --N 500 --steps 1000
  .\.venv312\Scripts\python scripts/compare_integrators.py --preset collision --N 1100 --steps 2000 --dt 0.005
Gera: data/energy_compare_<preset>.png + mostra interativo
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import matplotlib.pyplot as plt
import src.presets as presets
from src.presets_runner_fast import _get_backends
from src.nbody_sequential import SimConfig, SimResult
from src.visualization import plot_energy, plot_energy_comparison
import time, numpy as np

parser = argparse.ArgumentParser(description="Euler vs Leapfrog por preset (GPU)")
parser.add_argument("--preset", choices=["galaxy","collision","plummer","ring","triple","satellite","cold","kepler","disk","forming","solar"], default="galaxy")
parser.add_argument("--N", type=int, default=500)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--backend", type=str, default="gpu", help="gpu|numba|seq")
parser.add_argument("--outdir", type=str, default="data_final", help="pasta saída")
args = parser.parse_args()
pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)

def get_ic(preset, N, seed):
    if preset == "galaxy":
        return presets.generate_spiral_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=seed)
    elif preset == "collision":
        return presets.generate_galaxy_collision(seed=seed)
    elif preset == "plummer":
        pos, vel, masses = presets.generate_plummer_sphere(N, seed=seed)
        return pos, vel, masses, None
    elif preset == "ring":
        return presets.generate_ring_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=seed)
    elif preset == "triple":
        return presets.generate_triple_merger(N_per_galaxy=max(100,N//3), seed=seed)
    elif preset == "satellite":
        return presets.generate_satellite_infall(N_host_disk=int(N*0.7), N_host_bulge=N-int(N*0.7)-200, N_sat=200, seed=seed)
    elif preset == "cold":
        pos, vel, masses = presets.generate_cold_collapse(N, seed=seed)
        return pos, vel, masses, None
    elif preset == "kepler":
        pos, vel, masses = presets.generate_kepler_binary(seed=seed)
        return pos, vel, masses, None
    elif preset == "disk":
        pos, vel, masses = presets.generate_kepler_disk(N, seed=seed)
        return pos, vel, masses, None
    elif preset == "forming":
        return presets.generate_forming_galaxy(N=N, seed=seed)
    elif preset == "solar":
        return presets.generate_solar_system(seed=seed)
    else:
        raise ValueError(preset)

pos0, vel0, masses, labels = get_ic(args.preset, args.N, args.seed)
if labels is None:
    N = len(masses)
else:
    N = len(masses)
print(f"Preset {args.preset} N={N} steps={args.steps} dt={args.dt} backend={args.backend}")

compute_forces, total_energy, tag = _get_backends(args.backend)
print(f"Backend {args.backend} -> {tag}")

def run_with_integrator(integrator):
    pos = pos0.copy(); vel = vel0.copy()
    cfg = SimConfig(N=N, steps=args.steps, dt=args.dt, eps=args.eps, integrator=integrator, seed=args.seed)
    e0 = total_energy(pos, vel, masses, 1.0, args.eps)
    energy_history = [e0]
    if integrator == "leapfrog":
        accel = compute_forces(pos, masses, 1.0, args.eps)
    t0 = time.perf_counter()
    for step in range(1, args.steps+1):
        if integrator == "euler":
            accel = compute_forces(pos, masses, 1.0, args.eps)
            vel[:] = vel + accel * args.dt
            pos[:] = pos + vel * args.dt
        else:  # leapfrog
            vel_half = vel + accel * (args.dt*0.5)
            pos[:] = pos + vel_half * args.dt
            accel_new = compute_forces(pos, masses, 1.0, args.eps)
            vel[:] = vel_half + accel_new * (args.dt*0.5)
            accel[:] = accel_new
        if step % max(1, args.steps//200) == 0 or step == args.steps:
            e = total_energy(pos, vel, masses, 1.0, args.eps)
            energy_history.append(e)
    elapsed = time.perf_counter() - t0
    print(f"  {integrator} {elapsed:.2f}s  {elapsed/args.steps*1000:.2f} ms/step")
    return SimResult(pos=pos, vel=vel, masses=masses, config=cfg, elapsed=elapsed, energy_history=energy_history, trajectory=None)

results = {}
for integrator in ["leapfrog", "euler"]:
    print(f"\n=== {integrator} ===")
    res = run_with_integrator(integrator)
    if labels is not None:
        res.labels = labels  # type: ignore
    results[integrator] = res
    plot_energy(res, preset=args.preset, save_path=f"{args.outdir}/energy_{args.preset}_{integrator}.png")

print("\n=== Comparação Euler vs Leapfrog ===")
for k, res in results.items():
    e0 = res.energy_history[0][2]
    e1 = res.energy_history[-1][2]
    drift = abs(e1-e0)/max(1e-12, abs(e0))
    print(f"{k:8s} drift final {drift:.2e} {'OK' if drift<1e-2 else 'ALTO'}")

plot_energy_comparison(results, preset=args.preset, save_path=f"{args.outdir}/energy_compare_{args.preset}.png")
print(f"\nSalvos: {args.outdir}/energy_{args.preset}_leapfrog.png, {args.outdir}/energy_{args.preset}_euler.png, {args.outdir}/energy_compare_{args.preset}.png")
# não bloqueia em headless
try:
    plt.show()
except Exception:
    pass
