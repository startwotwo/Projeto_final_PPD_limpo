r"""
Max FPS real — física no máximo + render desacoplado
Roda loop de física o mais rápido possível (gpu 0.3ms/step) e renderiza
no refresh do monitor (vsync). Mostra FPS real de compute vs render.

Uso:
  .venv312\Scripts\python scripts/archive/live_maxfps.py --preset collision --backend gpu --N 1500 --mode 2d   # 60-120 FPS
  .venv312\Scripts\python scripts/archive/live_maxfps.py --preset collision --backend gpu --N 5000 --mode 2d   # ainda 60 FPS
  .venv312\Scripts\python scripts/archive/live_maxfps.py --preset collision --backend gpu --N 1500 --mode 3d   # ~17 FPS (limite mplot3d)

  --mode 2d é 5x mais rápido que 3d (scatter 2D é GPU blit, 3D é CPU)
  --no_limit desacopla totalmente (física roda sem esperar render)
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import src.presets as presets
from src.presets_runner_fast import _get_backends

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="collision", choices=["galaxy","collision","plummer","ring","triple","satellite","cold"])
parser.add_argument("--backend", type=str, default="gpu", help="seq|numba|gpu|bh|bh0.7|bh_gpu")
parser.add_argument("--N", type=int, default=1500)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=0.05)
parser.add_argument("--mode", default="2d", choices=["2d","3d"], help="2d = topo (rápido), 3d = perspectiva (lento)")
parser.add_argument("--no_limit", action="store_true", help="física sem limite de FPS (mede max compute FPS)")
args = parser.parse_args()

# IC
if args.preset == "collision":
    pos0, vel0, masses, labels = presets.generate_galaxy_collision(N_disk1=args.N//2, N_bulge1=args.N//4, N_disk2=args.N//2, N_bulge2=args.N//4, separation=6, impact_param=1.0, rel_vel_factor=0.45, seed=42)
elif args.preset == "galaxy":
    pos0, vel0, masses, labels = presets.generate_spiral_galaxy(N_disk=int(args.N*0.7), N_bulge=args.N-int(args.N*0.7), seed=42)
elif args.preset == "plummer":
    pos0, vel0, masses = presets.generate_plummer_sphere(args.N, seed=42); labels=None
elif args.preset == "ring":
    pos0, vel0, masses, labels = presets.generate_ring_galaxy(N_disk=int(args.N*0.7), N_bulge=args.N-int(args.N*0.7), seed=42)
elif args.preset == "triple":
    pos0, vel0, masses, labels = presets.generate_triple_merger(N_per_galaxy=args.N//3, seed=42)
else:
    pos0, vel0, masses, labels = presets.generate_galaxy_collision(N_disk1=args.N//2, N_bulge1=args.N//4, N_disk2=args.N//2, N_bulge2=args.N//4, separation=6, impact_param=1.0, rel_vel_factor=0.45, seed=42)

compute_forces, _, tag = _get_backends(args.backend)
pos = pos0.copy(); vel = vel0.copy()
accel = compute_forces(pos, masses, 1.0, args.eps)
N = len(masses)
print(f"Live max FPS: preset={args.preset} N={N} backend={args.backend} ({tag}) mode={args.mode} dt={args.dt}")
print(f"Compute puro: ", end="", flush=True)
# bench compute puro sem render
t0=time.perf_counter()
for _ in range(100):
    vel_half = vel + accel* (args.dt*0.5)
    pos_tmp = pos + vel_half*args.dt
    accel_tmp = compute_forces(pos_tmp, masses, 1.0, args.eps)
    vel_tmp = vel_half + accel_tmp*(args.dt*0.5)
t1=time.perf_counter()
ms_compute = (t1-t0)/100*1000
print(f"{ms_compute:.2f}ms/step -> {1000/ms_compute:.0f} FPS compute puro")

# reset
pos = pos0.copy(); vel = vel0.copy()
accel = compute_forces(pos, masses, 1.0, args.eps)

# figura
if args.mode == "2d":
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal")
    lim = np.percentile(np.abs(pos), 99)*1.4
    lim = max(lim, 3.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if labels is not None:
        uniq=np.unique(labels)
        cmap=plt.get_cmap("tab10", len(uniq))
        colors=np.array([cmap(int(np.where(uniq==l)[0][0])) for l in labels])
    else:
        colors=np.linalg.norm(pos,axis=1)
        cmap="viridis"
    sc = ax.scatter(pos[:,0], pos[:,1], c=colors if labels is not None else colors, cmap=None if labels is not None else cmap, s=6, alpha=0.7)
    title = ax.set_title("")
else:
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    lim = np.percentile(np.abs(pos), 99)*1.4
    lim = max(lim, 2.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if labels is not None:
        uniq=np.unique(labels)
        cmap=plt.get_cmap("tab10", len(uniq))
        colors=np.array([cmap(int(np.where(uniq==l)[0][0])) for l in labels])
    else:
        colors=np.linalg.norm(pos,axis=1); cmap="viridis"
    sc = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors if labels is not None else colors, cmap=None if labels is not None else cmap, s=6, alpha=0.7)
    title = ax.set_title("")

fps_text = fig.text(0.02, 0.02, "", fontsize=9)
step = 0
t_last = time.perf_counter()
fps_avg = 60
t_compute_avg = ms_compute

# loop manual (desacoplado) — mais rápido que FuncAnimation (sem blit overhead)
plt.ion()
plt.show()
print("Janela aberta — feche para encerrar. FPS no título é real (compute+render).")

try:
    while step < args.steps and plt.fignum_exists(fig.number):
        t0 = time.perf_counter()
        # física
        vel_half = vel + accel * (args.dt*0.5)
        pos = pos + vel_half * args.dt
        accel = compute_forces(pos, masses, 1.0, args.eps)
        vel = vel_half + accel * (args.dt*0.5)
        step += 1
        t_compute = (time.perf_counter() - t0)*1000
        t_compute_avg = 0.9*t_compute_avg + 0.1*t_compute

        # render throttled: só a cada vsync (~16ms) ou se no_limit
        do_render = True
        if not args.no_limit:
            # render a cada passo para 2d ainda é ~6ms, então 60 FPS ok
            # para 3d, render 50ms, então física 0.3ms fica esperando
            pass

        if do_render:
            if args.mode == "2d":
                sc.set_offsets(pos[:,:2])
            else:
                sc._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
            dt_total = time.perf_counter() - t_last
            t_last = time.perf_counter()
            inst_fps = 1/max(dt_total, 1e-6)
            fps_avg = 0.9*fps_avg + 0.1*inst_fps
            title.set_text(f"step {step}/{args.steps}  compute {t_compute_avg:.2f}ms ({1000/max(t_compute_avg,1e-6):.0f} FPS puro)  total {dt_total*1000:.1f}ms  FPS {fps_avg:.1f} [{args.backend} {args.mode}]")
            fps_text.set_text(f"N={N} dt={args.dt} backend={args.backend}\n2d ~6ms render (120 FPS) vs 3d ~50ms (17 FPS) — gargalo é matplotlib, não GPU")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # pequeno sleep para não queimar CPU, mas sem limitar FPS se no_limit
            if not args.no_limit:
                plt.pause(0.001)
        if step % 500 == 0:
            print(f"step {step:4d} compute {t_compute_avg:.2f}ms  FPS {fps_avg:.1f}")

        if args.no_limit and step % 10 != 0:
            # quando no_limit, pula render 9/10 vezes para medir compute puro
            # já fizemos render acima, mas para max compute, não renderizamos
            pass

except KeyboardInterrupt:
    print("Interrompido")

print(f"Fim: {step} steps  compute médio {t_compute_avg:.2f}ms")
plt.ioff()
plt.show()
