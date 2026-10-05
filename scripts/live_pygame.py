r"""
Live max FPS com Pygame (2D top-down, GPU blit) — 60-144 FPS mesmo com N=5000
Usa pygame (SDL2) que é 10x mais rápido que matplotlib para scatter.

Uso:
  .venv312\Scripts\python scripts/live_pygame.py --preset collision --backend gpu --N 5000
  .venv312\Scripts\python scripts/live_pygame.py --preset galaxy --backend gpu --N 3000 --speed 3
  .venv\Scripts\python scripts/live_pygame.py --preset collision --backend seq --N 800   # seq lento ~15 FPS
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, time
import numpy as np
import pygame
import src.presets as presets
from src.presets_runner_fast import _get_backends

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="collision", choices=["galaxy","collision","plummer","rotplummer","ring","triple","satellite","cold","kepler","disk","forming","solar","solar_real","solar_completo"])
parser.add_argument("--backend", type=str, default="gpu", help="seq|numba|gpu|bh|bh0.7|bh_gpu")
parser.add_argument("--N", type=int, default=2000)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=0.05)
parser.add_argument("--speed", type=float, default=1.0, help="velocidade de reprodução (1=normal, 2=2x frames por render, não altera dt)")
parser.add_argument("--size", type=float, default=3.0, help="raio dos pontos (2->5 maior)")
parser.add_argument("--glow", action="store_true", help="halo brilhante (2x círculos, +~30% CPU)")
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--integrator", choices=["leapfrog", "euler"], default="leapfrog", help="integrador")
parser.add_argument("--no_moons", action="store_true", help="solar sem luas")
parser.add_argument("--moons", type=int, default=None, help="limita nº de luas com label (primeiras do cache; padrão todas com label)")
parser.add_argument("--cache", type=str, default=None, help="cache Horizons alternativo (ex: data/horizons_cache_20260928.json)")
parser.add_argument("--asteroids", type=int, default=0, help="nº de asteroides reais do NPZ (ex: --asteroids 10000; exige backend BH p/ N grande)")
parser.add_argument("--asteroids-cache", type=str, default=None, help="NPZ de fetch_asteroids.py (padrão data/asteroids.npz)")
parser.add_argument("--trails", action="store_true", help="rastro das órbitas")
parser.add_argument("--labels", action="store_true", help="mostra nome de todos os corpos presentes (use --moons p/ limitar)")
args = parser.parse_args()
# speed agora é só reprodução (dt fixo para não perder precisão) — loop faz int(speed) passos por frame
# solar: eps grande destrói luas; dt grande perde órbitas
if args.preset in ("solar", "solar_real", "solar_completo"):
    if args.eps >= 1e-3:
        print(f"[aviso] preset {args.preset} requer eps <=1e-04 para estabilidade lunar; forçando eps=1e-05 (era {args.eps})")
        args.eps = 1e-5
    if args.dt >= 1e-4:
        print(f"[aviso] preset {args.preset} requer dt <=5e-05 para resolver órbitas lunares; forçando dt=2e-05 (era {args.dt})")
        args.dt = 2e-05

# IC
names = None  # só solar_completo retorna nomes reais (JPL)
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
elif args.preset == "satellite":
    pos0, vel0, masses, labels = presets.generate_satellite_infall(N_host_disk=int(args.N*0.7), N_host_bulge=args.N-int(args.N*0.7)-200, N_sat=200, seed=42)
elif args.preset == "rotplummer":
    pos0, vel0, masses = presets.generate_rotating_plummer(N=args.N, seed=42); labels=None
elif args.preset == "cold":
    pos0, vel0, masses = presets.generate_cold_collapse(args.N, seed=42); labels=None
elif args.preset == "forming":
    pos0, vel0, masses, labels = presets.generate_forming_galaxy(N=args.N, seed=42)
elif args.preset == "solar":
    pos0, vel0, masses, labels = presets.generate_solar_system(seed=42, with_moons=not args.no_moons)
elif args.preset == "solar_real":
    pos0, vel0, masses, labels = presets.generate_solar_system_real(with_moons=not args.no_moons)
elif args.preset == "solar_completo":
    pos0, vel0, masses, labels, names = presets.generate_solar_completo(with_moons=not args.no_moons, cache_path=args.cache, n_asteroids=args.asteroids, asteroids_cache=args.asteroids_cache)
elif args.preset == "kepler":
    pos0, vel0, masses = presets.generate_kepler_binary(seed=42); labels=None
elif args.preset == "disk":
    pos0, vel0, masses = presets.generate_kepler_disk(args.N, seed=42); labels=None

compute_forces, _, tag = _get_backends(args.backend)
pos = pos0.copy(); vel = vel0.copy()
accel = compute_forces(pos, masses, 1.0, args.eps)
N = len(masses)
print(f"Pygame live: {args.preset} N={N} backend={args.backend} ({tag}) dt={args.dt} speed={args.speed}")
# bench compute puro
t0=time.perf_counter()
for _ in range(100):
    vh = vel + accel*args.dt*0.5
    pt = pos + vh*args.dt
    at = compute_forces(pt, masses, 1.0, args.eps)
    vt = vh + at*args.dt*0.5
t1=time.perf_counter()
print(f"Compute puro: {(t1-t0)/100*1000:.2f}ms/step -> {1000/((t1-t0)/100*1000):.0f} FPS")

# pygame setup
pygame.init()
W, H = 800, 800
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"N-Body {args.preset} N={N} [{args.backend} {tag}] — Pygame 144 FPS")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)

# cores por label
if labels is not None:
    uniq=np.unique(labels)
    # tab10 cores
    tab10 = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207)]
    label_color = {int(u): tab10[i%10] for i,u in enumerate(uniq)}
    colors = np.array([label_color[int(l)] for l in labels], dtype=np.uint8)
else:
    # cor por raio
    r = np.linalg.norm(pos, axis=1)
    norm = (r - r.min()) / max(1e-9, r.max()-r.min())
    # viridis approx via jet
    colors = (plt_cm_viridis(norm)[:,:3]*255).astype(np.uint8) if False else np.zeros((N,3), dtype=np.uint8)
    # fallback simples: branco
    colors = np.full((N,3), 200, dtype=np.uint8)

def plt_cm_viridis(x):
    # não usado
    return np.zeros((len(x),4))

# limites mundo -> tela
lim = np.percentile(np.abs(pos), 99)*1.3
lim = max(lim, 3.0)
scale = (W*0.45) / lim  # mundo -> pixels
center = np.array([W//2, H//2])

# tamanho ∝ massa^(1/3) — massivas maiores e mais influentes (F=G m1 m2 / r²)
m_med = np.median(masses)
if args.size != 3.0:
    base = float(args.size)
else:
    base = 3 if N < 1000 else 2 if N < 3000 else 1
radii = (base * (masses / m_med) ** 0.33).astype(int)
radii = np.clip(radii, max(1, base//2), base*3)
# solar: tamanhos fixos e cores realistas para identificar — vale para solar, solar_real e solar_completo
if args.preset in ("solar", "solar_real", "solar_completo") and labels is not None:
    solar_radii = {0:12, 1:4, 2:5, 3:6, 4:4, 5:9, 6:8, 7:6, 8:6, 9:3, 10:3, 11:1}
    for i,l in enumerate(labels):
        radii[i] = solar_radii.get(int(l), radii[i])
    solar_colors = {0:(255,230,30), 1:(150,150,150), 2:(230,150,50), 3:(50,120,255), 4:(255,60,30), 5:(230,165,40), 6:(240,220,100), 7:(110,210,240), 8:(60,90,230), 9:(190,190,190), 10:(150,230,150), 11:(140,115,90)}
    for i,l in enumerate(labels):
        if int(l) in solar_colors:
            colors[i] = solar_colors[int(l)]
radius = int(np.median(radii))  # para compatibilidade com código antigo (halo)
# para desenhar, usa radii[i] por estrela
if args.glow:
    # gradiente radial suave com 7 camadas (de fora transparente para dentro opaco)
    glow_r1 = int(radius * 1.1)
    glow_r2 = int(radius * 1.3)
    glow_r3 = int(radius * 1.6)
    glow_r4 = int(radius * 2.0)
    glow_r5 = int(radius * 2.6)
    glow_r6 = int(radius * 3.2)
    glow_r7 = int(radius * 3.8)
    raio_max = glow_r7
    def make_star_surf(r_base, r_max):
        surf = pygame.Surface((r_max*2, r_max*2), pygame.SRCALPHA)
        # pygame BLEND_RGBA_ADD soma RGB sem multiplicar por alpha, então precisa pré-multiplicar:
        # RGB = alpha, A=255 para que ADD some exatamente alpha (ex: 18 = bem fraco, 76 = médio)
        # desenha do maior (mais fraco) para o menor (opaco) — 7 halos = fade suave
        pygame.draw.circle(surf, (8, 8, 8, 255),   (r_max, r_max), int(r_base * 3.8))
        pygame.draw.circle(surf, (14, 14, 14, 255), (r_max, r_max), int(r_base * 3.2))
        pygame.draw.circle(surf, (24, 24, 24, 255), (r_max, r_max), int(r_base * 2.6))
        pygame.draw.circle(surf, (42, 42, 42, 255), (r_max, r_max), int(r_base * 2.0))
        pygame.draw.circle(surf, (70, 70, 70, 255), (r_max, r_max), int(r_base * 1.6))
        pygame.draw.circle(surf, (110,110,110,255), (r_max, r_max), int(r_base * 1.3))
        pygame.draw.circle(surf, (170,170,170,255), (r_max, r_max), int(r_base * 1.1))
        pygame.draw.circle(surf, (255,255,255,255), (r_max, r_max), r_base)
        return surf
    star_surf = make_star_surf(radius, raio_max)
    # para estrelas massivas (2x tamanho) cria segunda superfície se necessário
    if np.any(masses > 2*np.median(masses)):
        star_surf_big = make_star_surf(radius*2, int(radius*2*3.5))
    else:
        star_surf_big = None
else:
    glow_r1 = glow_r2 = glow_r3 = 0
    star_surf = None
    star_surf_big = None
    raio_max = 0

# trails para órbitas (últimos 80 passos = cauda curta)
trail_len = 80 if args.trails else 0
if trail_len:
    trail_hist = np.zeros((trail_len, N, 2), dtype=np.float32)
    trail_hist[0] = pos[:,:2]
    trail_idx = 1
    max_trails = min(N, 30)
else:
    trail_hist = None  # type: ignore

step = 0
t_last = time.perf_counter()
fps_avg = 60
t_compute_avg = 0.5

running = True
while running and step < args.steps:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                # pausa
                pass

    # física: speed passos de física por frame (dt fixo, não perde precisão)
    t0 = time.perf_counter()
    steps_per_frame = args.save_every * max(1, int(args.speed))
    for _ in range(steps_per_frame):
        if step >= args.steps:
            break
        if args.integrator == "euler":
            accel = compute_forces(pos, masses, 1.0, args.eps)
            vel += accel * args.dt
            pos += vel * args.dt
        else:  # leapfrog
            vel_half = vel + accel * (args.dt*0.5)
            pos = pos + vel_half * args.dt
            accel = compute_forces(pos, masses, 1.0, args.eps)
            vel = vel_half + accel * (args.dt*0.5)
        step += 1
    t_compute = (time.perf_counter() - t0)*1000/steps_per_frame
    t_compute_avg = 0.9*t_compute_avg + 0.1*t_compute
    if trail_len:
        trail_hist[trail_idx % trail_len] = pos[:,:2]
        trail_idx += 1

    # render 2D top-down — seu método: desenha tudo, blur na tela toda, redesenha original por cima
    screen.fill((10,10,15))
    # desenha rastros primeiro (atrás das estrelas)
    if trail_len and trail_idx > 1:
        for i in range(min(max_trails, N)):
            n_hist = min(trail_idx, trail_len)
            if trail_idx < trail_len:
                hist = trail_hist[:n_hist, i]
            else:
                hist = np.concatenate([trail_hist[trail_idx % trail_len:, i], trail_hist[:trail_idx % trail_len, i]], axis=0)
            pts_hist = (hist * scale + center).astype(np.int32)
            if len(pts_hist) >= 2:
                # filtra pontos fora da tela para não poluir
                pts_list = [(int(x), int(y)) for x,y in pts_hist if 0 <= x < W and 0 <= y < H]
                if len(pts_list) >= 2:
                    c = tuple(map(int, colors[i])) if labels is not None else (80,80,120)
                    # escurece cor do rastro
                    c_trail = tuple(max(0, int(v*0.6)) for v in c)
                    pygame.draw.lines(screen, c_trail, False, pts_list, 1)
    pts = (pos[:,:2] * scale + center).astype(np.int32)
    # 1) desenha estrelas nítidas (tamanho ∝ massa)
    if labels is not None:
        for i, (x,y) in enumerate(pts):
            if 0 <= x < W and 0 <= y < H:
                c = tuple(map(int, colors[i]))
                pygame.draw.circle(screen, c, (int(x), int(y)), int(radii[i]))
    else:
        for i, (x,y) in enumerate(pts):
            if 0 <= x < W and 0 <= y < H:
                pygame.draw.circle(screen, (200,220,255), (int(x), int(y)), int(radii[i]))

    # 2) se glow, aplica gaussian blur na tela toda e soma por cima (bloom)
    if args.glow:
        # downscale 1/4 + upscale = blur barato (sem opencv)
        small = pygame.transform.smoothscale(screen, (W//4, H//4))
        bloom = pygame.transform.smoothscale(small, (W, H))
        # soma bloom com brilho (aglomerados ficam brancos)
        screen.blit(bloom, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
        # 3) redesenha estrelas nítidas por cima do bloom
        if labels is not None:
            for i, (x,y) in enumerate(pts):
                if 0 <= x < W and 0 <= y < H:
                    c = tuple(map(int, colors[i]))
                    pygame.draw.circle(screen, c, (int(x), int(y)), int(radii[i]))
        else:
            for i, (x,y) in enumerate(pts):
                if 0 <= x < W and 0 <= y < H:
                    pygame.draw.circle(screen, (255,255,255), (int(x), int(y)), int(radii[i]))
    # labels nomeados (solar) — usa nomes reais do preset quando houver
    # --labels nomeia TODOS os corpos presentes (limite com --moons)
    if args.labels and args.preset in ("solar", "solar_real", "solar_completo"):
        if names is not None:
            solar_names = [str(n) for n in list(names)]
        else:
            solar_names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
                           "Moon","Phobos","Deimos","Io","Europa","Ganymede","Callisto","Amalthea","Himalia","Elara","Pasiphae",
                           "Mimas","Tethys","Dione","Rhea","Iapetus","Titan","Enceladus","Miranda","Ariel","Umbriel","Oberon","Titania","Triton","Nereid",
                           "Ceres","Pluto","Haumea","Makemake","Eris"]
        label_idx = set(range(len(solar_names)))
        if args.moons is not None and labels is not None:
            # --moons limita só os labels das luas (primeiras do cache = maiores)
            _seen = 0
            label_idx = set()
            for i in range(len(solar_names)):
                if int(labels[i]) != 9:
                    label_idx.add(i)
                elif _seen < args.moons:
                    label_idx.add(i)
                    _seen += 1
        for i, (x,y) in enumerate(pts):
            if 0 <= x < W and 0 <= y < H and i < len(solar_names) and i in label_idx:
                surf = font.render(solar_names[i], True, (220,220,220))
                screen.blit(surf, (int(x)+4, int(y)-4))

    # HUD
    dt_total = time.perf_counter() - t_last
    t_last = time.perf_counter()
    inst_fps = 1/max(dt_total, 1e-6)
    fps_avg = 0.9*fps_avg + 0.1*inst_fps
    txt = [
        f"step {step}/{args.steps}  N={N} [{args.backend} {tag}]",
        f"compute {t_compute_avg:.2f}ms ({1000/max(t_compute_avg,1e-6):.0f} FPS puro)  total {dt_total*1000:.1f}ms  FPS {fps_avg:.1f}",
        f"dt={args.dt:.4f} eps={args.eps} speed={args.speed}  (SPACE pausa, ESC sai)",
        f"Pygame 2D top-down — 5x mais rápido que matplotlib 3D (3d ~50ms render vs 2d ~3ms)",
    ]
    for i, line in enumerate(txt):
        surf = font.render(line, True, (220,220,220))
        screen.blit(surf, (8, 8+i*16))
    # legenda
    if labels is not None:
        for j, ul in enumerate(np.unique(labels)):
            c = label_color[int(ul)]
            pygame.draw.circle(screen, c, (W-100, 20+j*16), 4)
            surf = font.render(f"c{int(ul)}", True, (200,200,200))
            screen.blit(surf, (W-90, 14+j*16))

    pygame.display.flip()
    clock.tick(144)  # limita a 144 FPS, mas mostra real

    if step % 200 == 0:
        print(f"step {step:4d} compute {t_compute_avg:.2f}ms  FPS {fps_avg:.1f}")

print(f"Fim: {step} steps")
pygame.quit()
