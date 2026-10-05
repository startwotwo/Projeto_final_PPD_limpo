r"""
Demonstração longa sem vídeo — usa GPU RTX 4060 Ti no máximo
Roda colisão completa com caudas de maré, 25 unidades de tempo (~4 órbitas)

Uso (Python 3.12 venv com GPU):
  .venv312\Scripts\python.exe scripts/run_long_gpu.py

Ou com .venv 3.14 (fallback CPU numba, ainda 150x mais rápido que seq):
  .venv\Scripts\python.exe scripts/run_long_gpu.py --backend numba

Gera apenas plots matplotlib interativos (sem mp4) — bem mais rápido.
"""
import pathlib, sys, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.presets import generate_galaxy_collision, generate_spiral_galaxy
from src.presets_runner_fast import run_fast
from src.visualization import plot_3d_snapshot, plot_energy, animate_trajectory

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="auto", choices=["auto","numba","gpu","cupy","torch","seq"])
parser.add_argument("--N", type=int, default=1200, help="N total por galáxia*2? Para colisão")
args = parser.parse_args()

if args.backend == "auto":
    try:
        import src.nbody_gpu as gpu
        args.backend = "gpu" if (gpu.cupy_available() or gpu.torch_available()) else "numba"
    except Exception:
        args.backend = "numba"

print(f"Backend: {args.backend} — RTX 4060 Ti" if args.backend=="gpu" else f"Backend: {args.backend}")

# 1) Galáxia isolada longa (para ver rotação estável)
print("\n=== Galáxia isolada longa (4 órbitas) ===")
pos, vel, masses, labels = generate_spiral_galaxy(N_disk=800, N_bulge=300, seed=42)
result = run_fast(pos, vel, masses, steps=3000, dt=0.005, backend=args.backend, save_every=5, verbose=True, labels=labels)
plot_3d_snapshot(result.pos, result.masses, labels=labels, title=f"Galáxia 1100 — 15 tempo [{args.backend}]")
plot_energy(result)
animate_trajectory(result, labels=labels, trails=True)  # interativo, sem salvar

# 2) Colisão longa com caudas de maré (Antennae) — 25 tempo
print("\n=== Colisão longa — caudas de maré (25 tempo) ===")
pos, vel, masses, labels = generate_galaxy_collision(
    N_disk1=600, N_bulge1=200, N_disk2=600, N_bulge2=200,
    separation=6.0, impact_param=1.2, rel_vel_factor=0.45,
    tilt1_deg=10, tilt2_deg=45, seed=123
)
print(f"N={len(masses)} sep=6 impact=1.2 — tempo total {4000*0.005} (4 órbitas externas)")
result2 = run_fast(pos, vel, masses, steps=4000, dt=0.005, backend=args.backend, save_every=8, verbose=True, labels=labels)
plot_3d_snapshot(result2.pos, result2.masses, labels=labels, title=f"Colisão 1600 — final 20 tempo [{args.backend}]")
plot_energy(result2)
animate_trajectory(result2, labels=labels, trails=True)

print("\nFim — feche as janelas matplotlib para encerrar")
