r"""
VisPy 3D Bloom — whole-scene blur (seu pipeline FBO) 
Por enquanto, VisPy fica sólido sem borda (60 FPS). Bloom tela cheia perfeito está em live_pygame.py --glow
Para VisPy bloom FBO completo, precisa view.draw() no FBO (ainda não estável, dá tela preta se FBO vazio).
Use live_vispy.py sem --glow para 3D sólido sem borda, ou live_pygame.py --glow para bloom.
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, time, numpy as np
import src.presets as presets
from src.presets_runner_fast import _get_backends
parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="collision", choices=["galaxy","collision","plummer","ring","triple","satellite","cold"])
parser.add_argument("--backend", type=str, default="gpu", help="seq|numba|gpu|bh")
parser.add_argument("--N", type=int, default=2000)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=0.05)
parser.add_argument("--size", type=float, default=6.0)
args = parser.parse_args()
if args.preset == "collision":
    pos0, vel0, masses, labels = presets.generate_galaxy_collision(N_disk1=args.N//2, N_bulge1=args.N//4, N_disk2=args.N//2, N_bulge2=args.N//4, separation=6, impact_param=1.0, rel_vel_factor=0.45, seed=42)
else:
    pos0, vel0, masses, labels = presets.generate_spiral_galaxy(N_disk=int(args.N*0.7), N_bulge=args.N-int(args.N*0.7), seed=42)
compute_forces, _, tag = _get_backends(args.backend)
pos = pos0.copy(); vel = vel0.copy()
accel = compute_forces(pos, masses, 1.0, args.eps)
N = len(masses)
print(f"VisPy Bloom: use live_pygame.py --glow para bloom tela cheia. VisPy 3D sólido N={N} [{args.backend}]")
from vispy import scene, app
canvas = scene.SceneCanvas(keys='interactive', show=True, size=(900,700), bgcolor="black")
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 12
lim = np.percentile(np.abs(pos), 99)*1.3
lim = max(lim, 3.0)
view.camera.center = (0,0,0)
scatter = scene.visuals.Markers(scaling='fixed', antialias=0)
if labels is not None:
    uniq=np.unique(labels)
    tab10=np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189]],dtype=float)/255
    colors=np.array([tab10[int(np.where(uniq==l)[0][0])%5] for l in labels], dtype=np.float32)
    colors=np.column_stack([colors, np.full(N,0.95)])
else:
    colors=np.ones((N,4), dtype=np.float32)
sizes = np.full(N, args.size, dtype=np.float32)
scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
view.add(scatter)
text = scene.visuals.Text("", color="white", font_size=9, pos=(10,20), parent=canvas.scene)
step=0; t_last=time.perf_counter(); fps_avg=60; t_compute_avg=0.5
accel_global=accel
def update(ev):
    global pos, vel, accel_global, step, t_last, fps_avg, t_compute_avg
    if step >= args.steps:
        app.quit()
        return
    t0=time.perf_counter()
    vel_half = vel + accel_global*(args.dt*0.5)
    pos[:] = pos + vel_half*args.dt
    accel_new = compute_forces(pos, masses, 1.0, args.eps)
    vel[:] = vel_half + accel_new*(args.dt*0.5)
    accel_global[:] = accel_new
    step+=1
    t_compute=(time.perf_counter()-t0)*1000
    scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
    dt_total=time.perf_counter()-t_last
    fps_avg=0.9*fps_avg+0.1*(1/max(dt_total,1e-6))
    canvas.title=f"step {step}/{args.steps}  FPS {fps_avg:.0f}  N={N} [{args.backend}]"
    text.text=f"FPS {fps_avg:.0f}  dt={args.dt:.4f}  N={N}  VisPy sólido sem borda — bloom use pygame"
canvas.events.draw.disconnect(canvas.on_draw)
canvas.events.draw.connect(lambda e: update(e))
timer = app.Timer(interval=0.0, connect=lambda e: canvas.update(), start=True)
if __name__ == "__main__":
    print("VisPy 3D sólido sem borda preta. Para bloom tela cheia use: live_pygame.py --glow")
    app.run()
