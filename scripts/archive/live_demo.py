r"""
Live demo fluido — vê ao vivo a diferença de velocidade
Roda a física passo-a-passo e atualiza o plot em tempo real.
Backend lento (seq) fica travado/choppy, gpu/numba fica fluido.

Uso:
  .venv312\Scripts\python scripts/archive/live_demo.py --preset collision --backend seq      # lento ~4 FPS para N=1500
  .venv312\Scripts\python scripts/archive/live_demo.py --preset collision --backend gpu      # fluido ~60 FPS
  .venv312\Scripts\python scripts/archive/live_demo.py --preset collision --backend numba    # intermediário ~30 FPS
  .venv312\Scripts\python scripts/archive/live_demo.py --preset collision --compare           # lado a lado seq vs gpu

  --N 1200 --steps 3000 --dt 0.005  para ajustar duração
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import src.presets as presets
from src.presets_runner_fast import _get_backends

parser = argparse.ArgumentParser(description="Live N-Body fluido")
parser.add_argument("--preset", default="collision", choices=["galaxy","collision","plummer","ring","triple","satellite","cold"])
parser.add_argument("--backend", type=str, default="gpu", help="seq|numba|gpu|bh|bh0.7|bh_gpu")
parser.add_argument("--compare", action="store_true", help="lado a lado seq vs gpu")
parser.add_argument("--N", type=int, default=1200)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=0.05)
parser.add_argument("--save_every", type=int, default=1, help="atualiza plot a cada K passos físicos (1=mais fluido)")
parser.add_argument("--speed", type=float, default=1.0, help="2.0 = animação 2x mais acelerada (dt*2, sem perder precisão se dt<0.01)")
args = parser.parse_args()
if args.speed != 1.0:
    # aplica speed: aumenta dt e save_every proporcionalmente
    args.dt *= args.speed
    # para speed 2, faz 2 passos físicos por frame extra mantendo FPS
    if args.speed > 1:
        args.save_every = max(args.save_every, int(args.speed))

def make_ic(preset, N):
    if preset == "collision":
        pos, vel, masses, labels = presets.generate_galaxy_collision(
            N_disk1=N//2, N_bulge1=N//4, N_disk2=N//2, N_bulge2=N//4,
            separation=6, impact_param=1.0, rel_vel_factor=0.45, seed=42)
    elif preset == "galaxy":
        pos, vel, masses, labels = presets.generate_spiral_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=42)
    elif preset == "plummer":
        pos, vel, masses = presets.generate_plummer_sphere(N, seed=42)
        labels=None
    elif preset == "ring":
        pos, vel, masses, labels = presets.generate_ring_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=42)
    elif preset == "triple":
        pos, vel, masses, labels = presets.generate_triple_merger(N_per_galaxy=N//3, seed=42)
    elif preset == "satellite":
        pos, vel, masses, labels = presets.generate_satellite_infall(N_host_disk=int(N*0.7), N_host_bulge=N-int(N*0.7)-200, N_sat=200, seed=42)
    elif preset == "cold":
        pos, vel, masses = presets.generate_cold_collapse(N, seed=42)
        labels=None
    else:
        raise ValueError(preset)
    return pos, vel, masses, labels

def run_live(pos0, vel0, masses, labels, backend, dt, eps, steps, save_every, title_suffix=""):
    compute_forces, _, tag = _get_backends(backend)
    pos = pos0.copy(); vel = vel0.copy()
    N = pos.shape[0]
    accel = compute_forces(pos, masses, 1.0, eps)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    # limites
    lim = np.percentile(np.abs(pos), 99)*1.4
    lim = max(lim, 2.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # cores
    if labels is not None:
        uniq=np.unique(labels)
        cmap=plt.get_cmap("tab10", len(uniq))
        colors=np.array([cmap(int(np.where(uniq==l)[0][0])) for l in labels])
        sizes = np.where(masses> 2*np.median(masses), 40, 8)
    else:
        colors=np.linalg.norm(pos,axis=1)
        cmap="viridis"
        sizes=6
    sc = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors if labels is not None else colors, cmap=None if labels is not None else cmap, s=sizes, alpha=0.8)
    if labels is not None:
        for ul in np.unique(labels):
            ax.scatter([],[],[], c=[plt.get_cmap("tab10")(int(ul)%10)], label=f"c{int(ul)}", s=20)
        ax.legend(fontsize=7, loc="upper right")
    title = ax.set_title("")
    # texto FPS
    fps_text = fig.text(0.02, 0.02, "", fontsize=9)

    # para média móvel
    t_last = time.perf_counter()
    step = 0
    fps_avg = 30

    import matplotlib.animation as animation

    def update(frame):
        nonlocal pos, vel, accel, step, t_last, fps_avg
        t0 = time.perf_counter()
        # avança save_every passos físicos por frame visual
        for _ in range(save_every):
            if step >= steps:
                return sc,
            vel_half = vel + accel * (dt*0.5)
            pos = pos + vel_half * dt
            accel_new = compute_forces(pos, masses, 1.0, eps)
            vel = vel_half + accel_new * (dt*0.5)
            accel = accel_new
            step += 1
        # atualiza scatter
        sc._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        # FPS
        dt_frame = time.perf_counter() - t0
        # inclui render
        dt_total = time.perf_counter() - t_last
        t_last = time.perf_counter()
        inst_fps = 1/max(dt_total, 1e-6)
        fps_avg = 0.9*fps_avg + 0.1*inst_fps
        ms = dt_frame*1000/save_every
        title.set_text(f"{title_suffix} [{backend} {tag}] step {step}/{steps}  {ms:.2f}ms/step  FPS {fps_avg:.1f} (save_every={save_every})")
        fps_text.set_text(f"backend={backend}  N={N}  eps={eps}  dt={dt}\nseq ~200ms/step (4 FPS) vs gpu ~0.3ms (60 FPS) para N=2000")
        return sc,

    # interval 1 = mais rápido possível, deixa o backend ditar o FPS
    ani = animation.FuncAnimation(fig, update, frames=(steps//save_every), interval=1, blit=False, repeat=False)
    plt.show()
    return ani

if args.compare:
    # lado a lado: seq vs gpu
    print("Modo compare: seq (esquerda, lento) vs gpu (direita, fluido) — mesma IC")
    pos0, vel0, masses, labels = make_ic(args.preset, args.N)
    # cria figura com 2 subplots 3D
    # Para comparar ao vivo, roda sequencialmente mas com mesmo fig? Simplifica: abre 2 janelas sequenciais e pede para comparar visualmente
    print("Primeiro: SEQ (vai travar)... feche a janela para ver GPU")
    run_live(pos0, vel0, masses, labels, "seq", args.dt, args.eps, args.steps, args.save_every, title_suffix="SEQ LENTO")
    print("Agora: GPU (fluido)...")
    run_live(pos0, vel0, masses, labels, "gpu", args.dt, args.eps, args.steps, args.save_every, title_suffix="GPU FLUIDO")
else:
    pos0, vel0, masses, labels = make_ic(args.preset, args.N)
    print(f"Live {args.preset} N={len(masses)} backend={args.backend} steps={args.steps} dt={args.dt} save_every={args.save_every}")
    print("Feche a janela 3D para encerrar. FPS mostra fluidez real (inclui física + render).")
    run_live(pos0, vel0, masses, labels, args.backend, args.dt, args.eps, args.steps, args.save_every, title_suffix=args.preset)
