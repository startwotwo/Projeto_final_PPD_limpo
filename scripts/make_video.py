"""Gera vídeo local a partir da simulação sequencial (para entrega ou Colab fallback)."""
import pathlib
import sys
# garante que `src` seja encontrável tanto como `src.` quanto como módulo top-level
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.nbody_sequential import SimConfig, run_simulation
from src.visualization import plot_3d_snapshot, plot_energy, animate_trajectory

cfg = SimConfig(N=500, steps=300, dt=1e-4, integrator="leapfrog", seed=42)
result = run_simulation(cfg, save_trajectory=True, energy_every=10, verbose=True)

# snapshots
plot_3d_snapshot(result.pos, result.masses, save_path="data/snapshot_final.png")
plot_energy(result, save_path="data/energia.png")

# vídeo — tenta mp4, fallback gif
animate_trajectory(result, save_path="data/nbody_fase1.mp4", fps=25)
print("Done. Veja data/nbody_fase1.mp4 (ou .gif se ffmpeg ausente)")
