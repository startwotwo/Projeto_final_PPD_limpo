"""
Runner de presets elaborados — galáxias, colisões, órbitas, discos
Uso:
  .venv/Scripts/python scripts/run_presets.py --preset galaxy --steps 1500 --dt 0.002
  .venv/Scripts/python scripts/run_presets.py --preset collision --steps 2000
  .venv/Scripts/python scripts/run_presets.py --preset kepler --ecc 0.6
  .venv/Scripts/python scripts/run_presets.py --preset disk --N 500
  .venv/Scripts/python scripts/run_presets.py --preset plummer

Gera mp4 em data/ com ffmpeg (já instalado).
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import matplotlib
# não força Agg aqui — deixa visualizar; para salvar usa ffmpeg

import src.presets as presets
from src.presets_runner import run_custom_simulation
from src.presets_runner_fast import run_fast
from src.visualization import plot_3d_snapshot, plot_2d_projections, plot_energy, animate_trajectory

def preset_galaxy(args):
    # Coerência com live_vispy.py: se --N foi passado (diferente do default 800), usa N*0.7/N*0.3
    if args.N != 800:
        N_disk = int(args.N*0.7); N_bulge = args.N - N_disk
    else:
        N_disk = args.N_disk; N_bulge = args.N_bulge
    pos, vel, masses, labels = presets.generate_spiral_galaxy(
        N_disk=N_disk, N_bulge=N_bulge, M_total=1.0, R_d=1.0, a_bulge=0.4,
        z_d=0.08, bulge_frac=0.25, G=1.0, eps=args.eps, seed=args.seed,
        with_bh=args.bh, bh_frac=0.03
    )
    return pos, vel, masses, labels, f"Galáxia espiral N={len(masses)} (disk {N_disk}+bulge {N_bulge}{' +BH' if args.bh else ''})"

def preset_collision(args):
    if args.N != 800:
        N_disk = args.N//2; N_bulge = args.N//4
        N_disk1=N_disk; N_bulge1=N_bulge; N_disk2=N_disk; N_bulge2=N_bulge
    else:
        N_disk1=args.N_disk//2; N_bulge1=args.N_bulge//2; N_disk2=args.N_disk//2; N_bulge2=args.N_bulge//2
    pos, vel, masses, labels = presets.generate_galaxy_collision(
        N_disk1=N_disk1, N_bulge1=N_bulge1,
        N_disk2=N_disk2, N_bulge2=N_bulge2,
        M_total1=1.0, M_total2=1.0, R_d=1.0, separation=args.sep,
        impact_param=args.impact, rel_vel_factor=args.vfac,
        tilt1_deg=0, tilt2_deg=35, G=1.0, eps=args.eps, seed=args.seed, with_bh=args.bh
    )
    return pos, vel, masses, labels, f"Colisão N={len(masses)} sep={args.sep} impact={args.impact} vfac={args.vfac}"

def preset_plummer(args):
    pos, vel, masses = presets.generate_plummer_sphere(N=args.N, M_total=1.0, a=0.6, G=1.0, seed=args.seed)
    labels = None
    return pos, vel, masses, labels, f"Plummer N={args.N} a=0.6 (virializado)"

def preset_kepler(args):
    pos, vel, masses = presets.generate_kepler_binary(M1=0.5, M2=0.5, separation=1.0, eccentricity=args.ecc, G=1.0)
    labels = np.array([0, 1], dtype=np.int32)
    return pos, vel, masses, labels, f"Binária Kepler e={args.ecc} a=1.0"

def preset_disk(args):
    pos, vel, masses = presets.generate_kepler_disk(N=args.N, M_central=1.0, m_particle=1e-4, R_min=0.6, R_max=2.2, eccentricity=args.ecc, G=1.0, eps=args.eps, seed=args.seed)
    labels = np.zeros(len(masses), dtype=np.int32)
    labels[0] = 1  # central
    return pos, vel, masses, labels, f"Disco Kepleriano N={args.N} R=[0.6,2.2] e<{args.ecc}"

def preset_cold(args):
    pos, vel, masses = presets.generate_cold_collapse(N=args.N, radius=1.5, seed=args.seed)
    return pos, vel, masses, None, f"Cold collapse N={args.N} R=1.5 (quedas livre)"

def preset_rotplummer(args):
    pos, vel, masses = presets.generate_rotating_plummer(N=args.N, spin=0.5, seed=args.seed)
    return pos, vel, masses, None, f"Plummer rotante N={args.N} spin=0.5"

def preset_ring(args):
    if args.N != 800:
        N_disk=int(args.N*0.7); N_bulge=args.N-N_disk
    else:
        N_disk=args.N_disk; N_bulge=args.N_bulge
    pos, vel, masses, labels = presets.generate_ring_galaxy(N_disk=N_disk, N_bulge=N_bulge, seed=args.seed, eps=args.eps)
    return pos, vel, masses, labels, f"Cartwheel (anel) N={len(masses)}"

def preset_triple(args):
    N_per = args.N//3 if args.N != 800 else args.N//3
    pos, vel, masses, labels = presets.generate_triple_merger(N_per_galaxy=N_per, separation=args.sep, seed=args.seed, eps=args.eps)
    return pos, vel, masses, labels, f"Triple merger 3x{N_per} N={len(masses)}"

def preset_satellite(args):
    if args.N != 800:
        N_disk=int((args.N-200)*0.7); N_bulge=args.N-200-N_disk
    else:
        N_disk=args.N_disk; N_bulge=args.N_bulge
    pos, vel, masses, labels = presets.generate_satellite_infall(N_host_disk=N_disk, N_host_bulge=N_bulge, N_sat=200, seed=args.seed, eps=args.eps)
    return pos, vel, masses, labels, f"Satélite infall N={len(masses)}"

PRESETS = {
    "galaxy": preset_galaxy,
    "collision": preset_collision,
    "plummer": preset_plummer,
    "kepler": preset_kepler,
    "disk": preset_disk,
    "cold": preset_cold,
    "rotplummer": preset_rotplummer,
    "ring": preset_ring,
    "triple": preset_triple,
    "satellite": preset_satellite,
}

def main():
    p = argparse.ArgumentParser(description="Presets N-Body elaborados")
    p.add_argument("--preset", choices=PRESETS.keys(), default="galaxy", help="qual simulação")
    p.add_argument("--N", type=int, default=800, help="N para plummer/disk")
    p.add_argument("--N_disk", type=int, default=700)
    p.add_argument("--N_bulge", type=int, default=300)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sep", type=float, default=5.0, help="separação colisão")
    p.add_argument("--impact", type=float, default=0.5)
    p.add_argument("--vfac", type=float, default=0.5, help="fator velocidade relativa (1=parabólica)")
    p.add_argument("--ecc", type=float, default=0.6, help="excentricidade kepler/disk")
    p.add_argument("--bh", action="store_true", help="inclui buraco negro central")
    p.add_argument("--save_every", type=int, default=3, help="salva 1 a cada K steps")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no_video", action="store_true", help="não gera mp4, só mostra interativo (RECOMENDADO para N>1000, muito mais rápido)")
    p.add_argument("--video", action="store_true", help="força geração de mp4 mesmo para N grande (lento)")
    p.add_argument("--backend", type=str, default="auto", help="backend: seq|numba|gpu|cupy|torch|tiled|bh|bh0.5|bh0.7|bh0.9 (auto=GPU se disponível)")
    p.add_argument("--no_plot", action="store_true", help="não mostra plots (só salva png)")
    args = p.parse_args()

    # auto-detect backend
    if args.backend == "auto":
        try:
            import src.nbody_gpu as gpu
            if gpu.cupy_available() or gpu.torch_available():
                args.backend = "gpu"
            else:
                args.backend = "numba"
        except Exception:
            args.backend = "numba"

    pos, vel, masses, labels, title = PRESETS[args.preset](args)
    print(f"\n=== {title} ===")
    print(f"pos range: {pos.min():.2f}..{pos.max():.2f}  vel rms: {np.sqrt(np.mean(np.sum(vel*vel,axis=1))):.3f}  Mtot={masses.sum():.2f}")
    print(f"backend: {args.backend}")

    if args.backend.startswith("bh") or args.backend in ("numba", "gpu", "cupy", "torch", "cupy_fast"):
        result = run_fast(
            pos, vel, masses, steps=args.steps, dt=args.dt, G=1.0, eps=args.eps,
            backend=args.backend, save_trajectory=True, save_every=args.save_every,
            energy_every=max(10, args.steps//20), verbose=True, labels=labels
        )
    else:
        tiled = len(masses) > 1500 or args.backend == "tiled"
        result = run_custom_simulation(
            pos, vel, masses, steps=args.steps, dt=args.dt, G=1.0, eps=args.eps,
            integrator="leapfrog", tiled=tiled, save_trajectory=True, save_every=args.save_every,
            energy_every=max(10, args.steps//20), verbose=True, labels=labels
        )

    # Apenas métricas no console quando --no_plot (usuário usa só livevispy)
    tag = f"{args.preset}_N{len(masses)}_steps{args.steps}_dt{args.dt}_{args.backend}"
    if not args.no_plot:
        if labels is not None:
            plot_3d_snapshot(result.pos, result.masses, title=f"{title} — final [{args.backend}]", labels=labels)
            plot_2d_projections(result.pos, labels=labels, title=f"{title} — projeções")
        else:
            plot_3d_snapshot(result.pos, result.masses, title=f"{title} — final [{args.backend}]")
        plot_energy(result)
        animate_trajectory(result, save_path=None, labels=labels, trails=(args.preset in ("collision","galaxy","ring","triple","satellite")))
    else:
        print(f"(sem plots — use sem --no_plot para ver snap/proj/energy)")

    # Vídeo removido para este projeto (só livevispy)
    print("Vídeo desativado para este projeto (use live_vispy.py --trails --glow ao vivo)")

    # drift final
    e0 = result.energy_history[0][2]
    e1 = result.energy_history[-1][2]
    drift = abs(e1-e0)/max(1e-12, abs(e0))
    print(f"Drift energia: {drift:.2e} {'OK' if drift<0.05 else 'ALTO - reduza dt'}")

if __name__ == "__main__":
    main()
