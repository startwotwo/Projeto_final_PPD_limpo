"""
Gera todos os vídeos elaborados de uma vez — galeria completa para entrega.
Roda presets com parâmetros bonitos e estáveis, salva mp4 em data/gallery_*

Uso: .venv/Scripts/python scripts/make_all_presets.py
Tempo total ~2-3 min para N~400-600.
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.presets import (
    generate_spiral_galaxy, generate_galaxy_collision,
    generate_plummer_sphere, generate_kepler_binary, generate_kepler_disk
)
from src.presets_runner import run_custom_simulation
from src.visualization import plot_3d_snapshot, plot_2d_projections, plot_energy, animate_trajectory
import numpy as np

def run_and_save(name, pos, vel, masses, labels, steps, dt, eps, save_every=3, fps=30, trails=False):
    print(f"\n{'='*60}\n{name}  N={len(masses)} steps={steps} dt={dt}\n{'='*60}")
    result = run_custom_simulation(pos, vel, masses, steps=steps, dt=dt, G=1.0, eps=eps,
                                   tiled=len(masses)>1500, save_trajectory=True, save_every=save_every,
                                   energy_every=max(10, steps//20), verbose=True, labels=labels)
    tag = f"gallery_{name}"
    plot_3d_snapshot(result.pos, result.masses, title=f"{name} — final", save_path=f"data/{tag}_snap.png", labels=labels)
    if labels is not None and len(np.unique(labels))>1:
        plot_2d_projections(result.pos, labels=labels, title=f"{name} — projeções", save_path=f"data/{tag}_proj.png")
    plot_energy(result, save_path=f"data/{tag}_energy.png")
    animate_trajectory(result, save_path=f"data/{tag}.mp4", fps=fps, labels=labels, trails=trails)
    return result

# 1) Órbita Kepler binária excêntrica (referência visual simples)
pos, vel, masses = generate_kepler_binary(M1=0.5, M2=0.5, separation=1.0, eccentricity=0.6)
run_and_save("01_kepler_e06", pos, vel, masses, np.array([0,1],dtype=np.int32), steps=900, dt=0.005, eps=0.01, save_every=2, fps=30)

pos, vel, masses = generate_kepler_binary(M1=0.5, M2=0.5, separation=1.0, eccentricity=0.0)
run_and_save("02_kepler_circular", pos, vel, masses, np.array([0,1],dtype=np.int32), steps=900, dt=0.005, eps=0.01, save_every=2, fps=30)

# 2) Plummer estável (aglomerado globular)
pos, vel, masses = generate_plummer_sphere(N=600, M_total=1.0, a=0.6, seed=42)
run_and_save("03_plummer_600", pos, vel, masses, None, steps=800, dt=0.005, eps=0.03, save_every=3, fps=30)

# 3) Galáxia espiral isolada (disco + bulbo, rotação realista)
pos, vel, masses, labels = generate_spiral_galaxy(N_disk=700, N_bulge=300, M_total=1.0, R_d=1.0, a_bulge=0.4, seed=42)
run_and_save("04_galaxy_1000", pos, vel, masses, labels, steps=1200, dt=0.003, eps=0.05, save_every=3, fps=30, trails=True)

# 4) Galáxia com BH central (destaca massa)
pos, vel, masses, labels = generate_spiral_galaxy(N_disk=600, N_bulge=200, M_total=1.0, R_d=1.0, a_bulge=0.4, seed=7, with_bh=True, bh_frac=0.05)
run_and_save("05_galaxy_BH", pos, vel, masses, labels, steps=1000, dt=0.003, eps=0.03, save_every=3, fps=30, trails=True)

# 5) Disco Kepleriano (sistema planetário)
pos, vel, masses = generate_kepler_disk(N=600, M_central=1.0, m_particle=1e-4, R_min=0.6, R_max=2.2, eccentricity=0.15, seed=42)
labels = np.zeros(len(masses), dtype=np.int32); labels[0]=1
run_and_save("06_kepler_disk_600", pos, vel, masses, labels, steps=1200, dt=0.002, eps=0.02, save_every=3, fps=30)

# 6) Colisão frontal (impacto pequeno, fusão rápida)
pos, vel, masses, labels = generate_galaxy_collision(N_disk1=400, N_bulge1=150, N_disk2=400, N_bulge2=150,
                                                     separation=5.0, impact_param=0.3, rel_vel_factor=0.6, tilt1_deg=0, tilt2_deg=10, seed=42)
run_and_save("07_collision_frontal", pos, vel, masses, labels, steps=1400, dt=0.003, eps=0.05, save_every=3, fps=30, trails=True)

# 7) Colisão com parâmetro de impacto grande + inclinação (gera caudas de maré — Antennae)
pos, vel, masses, labels = generate_galaxy_collision(N_disk1=400, N_bulge1=150, N_disk2=400, N_bulge2=150,
                                                     separation=6.0, impact_param=1.2, rel_vel_factor=0.45, tilt1_deg=10, tilt2_deg=45, seed=123)
run_and_save("08_collision_tidal", pos, vel, masses, labels, steps=1600, dt=0.003, eps=0.05, save_every=3, fps=30, trails=True)

print("\nTodos os vídeos em data/gallery_*.mp4")
