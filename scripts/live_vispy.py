import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, time, numpy as np
import src.presets as presets
from src.presets_runner_fast import _get_backends

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="collision", choices=["galaxy","collision","plummer","rotplummer","ring","triple","satellite","cold","kepler","disk","forming","solar","solar_real","solar_completo"])
parser.add_argument("--glow", action="store_true", help="bloom whole-scene (captura + PIL blur + ADD)")
parser.add_argument("--backend", type=str, default="gpu", help="seq|numba|gpu|bh")
parser.add_argument("--N", type=int, default=2000)
parser.add_argument("--steps", type=int, default=50000)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--eps", type=float, default=0.05)
parser.add_argument("--size", type=float, default=6.0)
parser.add_argument("--speed", type=float, default=1.0, help="velocidade de reprodução (1=normal, 2=2x frames por render, não altera dt)")
parser.add_argument("--integrator", choices=["leapfrog", "euler"], default="leapfrog", help="integrador")
parser.add_argument("--no_moons", action="store_true", help="solar sem luas (só Sol+8 planetas)")
parser.add_argument("--moons", type=int, default=None, help="limita nº de luas com label (primeiras do cache; padrão todas com label)")
parser.add_argument("--cache", type=str, default=None, help="cache Horizons alternativo (ex: data/horizons_cache_20260928.json)")
parser.add_argument("--asteroids", type=int, default=0, help="nº de asteroides reais do NPZ (ex: --asteroids 10000; exige backend BH p/ N grande)")
parser.add_argument("--asteroids-cache", type=str, default=None, help="NPZ de fetch_asteroids.py (padrão data/asteroids.npz)")
parser.add_argument("--trails", nargs="?", const="40", default=None, metavar="N", help="rastro: número de pontos ou 'inf' para nunca apagar (ex: --trails 200, --trails inf, --trails 500)")
parser.add_argument("--labels", action="store_true", help="mostra nome de todos os corpos presentes (use --moons p/ limitar)")
parser.add_argument("--label-size", type=float, default=0.01, help="tamanho da fonte dos labels")
parser.add_argument("--label-dz", type=float, default=0.0, help="offset em z do label em relação ao corpo (0 = colado)")
parser.add_argument("--adaptive", action="store_true", help="dt adaptativo para solar (0.002 longe -> 0.0002 perto)")
parser.add_argument("--follow", type=str, default=None, help="ancora câmera no corpo (ex: Earth, Jupiter, Sun) — Shift+drag desancora")
parser.add_argument("--debug-labels", action="store_true", help="imprime pos mundo corpo vs label e salva debug_labels.png no step 30")
parser.add_argument("--hold", action="store_true", help="não fecha no fim dos steps (congela no estado final)")
parser.add_argument("--sunray", action="store_true", help="raio amarelo Sol->Terra (direção da luz, presets solar)")
parser.add_argument("--topdown", action="store_true", help="câmera travada de cima da eclíptica (compara forma sem paralaxe)")
parser.add_argument("--until", type=str, default=None, help="simula até a data/hora e congela (ex: 2026-09-29, '2026-09-26 16:49', 'now'); só solar_completo com cache")
parser.add_argument("--record", type=str, default=None, metavar="OUT.mp4", help="grava vídeo via ffmpeg (ex: --record data/videos/galaxy.mp4)")
parser.add_argument("--record-fps", type=float, default=30.0, help="fps do vídeo gravado")
parser.add_argument("--record-every", type=int, default=1, help="captura 1 frame a cada K frames")
parser.add_argument("--camera-orbit", action="store_true", help="câmera orbita sozinha durante a execução (azimute varre + elevação oscila até 0° e volta)")
parser.add_argument("--orbit-az", type=float, default=150.0, help="varredura horizontal total em graus ao longo do vídeo")
parser.add_argument("--orbit-el-cycles", type=float, default=2.0, help="nº de oscilações verticais (início→0°→início) ao longo do vídeo")
parser.add_argument("--axes", action="store_true", help="mostra eixos XYZ fixos no mundo (referencial inercial: distingue movimento de câmera vs. dos corpos)")
parser.add_argument("--axes-len", type=float, default=0.0, help="comprimento dos eixos (0 = escala automática da cena)")
args = parser.parse_args()

# solar: garante eps pequeno e dt pequeno para luas não espiralarem (a_lua ~0.0003, eps 0.05>>a destrói; dt 0.005>> período Io 0.001)
if args.preset in ("solar", "solar_real", "solar_completo"):
    if args.eps >= 1e-3:
        print(f"[aviso] preset {args.preset} requer eps <=1e-04 para estabilidade lunar; forçando eps=1e-05 (era {args.eps})")
        args.eps = 1e-5
    if args.dt >= 1e-4:
        print(f"[aviso] preset {args.preset} requer dt <=5e-05 para resolver órbitas lunares (período Io ~0.001); forçando dt=2e-05 (era {args.dt}) e adaptive=True")
        args.dt = 2e-05
        args.adaptive = True

# --until DATA: calcula steps p/ chegar na data partindo da época do cache e congela lá
# Aceita dia (2026-09-29), ISO com hora ('2026-09-26 16:49') ou com traços (2026-09-26-16-49)
def _parse_until(s):
    import re as _re, datetime as _dt2
    s = s.strip()
    if s.lower() == "now":
        return _dt2.datetime.now()
    s = _re.sub(r"^(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})$", r"\1 \2:\3", s)
    return _dt2.datetime.fromisoformat(s)

if args.until:
    if args.preset != "solar_completo":
        raise SystemExit("--until só funciona com --preset solar_completo (precisa da época do cache)")
    import json as _json, datetime as _dt, math as _math
    _cp = pathlib.Path(args.cache) if args.cache else pathlib.Path("data/horizons_cache.json")
    _epoch = _json.loads(_cp.read_text(encoding="utf-8")).get("epoch", "2026-01-01")
    _days = (_parse_until(args.until) - _dt.datetime.fromisoformat(_epoch)).total_seconds() / 86400.0
    if _days <= 0:
        raise SystemExit(f"--until {args.until} é anterior à época do cache ({_epoch})")
    _T = _math.sqrt(1.0 * 0.998 * 1000.0 / (4 * _math.pi ** 2))  # G=1.0, M_sol=0.998 (mesma escala do preset)
    _t_sim = (_days / 365.25) / _T
    _per_frame = max(1, int(args.speed))  # cada frame integra `speed` substeps de dt
    args.steps = max(1, int(round(_t_sim / (args.dt * _per_frame))))
    args.hold = True
    # dt FIXO: com adaptive o dt real encolhe (Phobos domina max_a) e steps*dt
    # deixaria de valer — a simulação pararia meses antes do alvo
    args.adaptive = False
    args._until_epoch = _epoch
    args._until_T = _T
    print(f"Alvo: {_epoch} -> {args.until} ({_days:.2f} dias = {_t_sim:.6f} t.sim): {args.steps} steps dt={args.dt} FIXO (adaptive off), congela na chegada")

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
elif args.preset == "kepler": 
    pos0, vel0, masses = presets.generate_kepler_binary(M1 = 1, M2 = 1, separation = 1, eccentricity = 0.7, G = 1, seed=42); labels=None
elif args.preset == "disk":
    pos0, vel0, masses = presets.generate_kepler_disk(args.N, seed=42); labels=None
elif args.preset == "forming":
    pos0, vel0, masses, labels = presets.generate_forming_galaxy(N=args.N, radius=1.0, spin=1, seed=42)
elif args.preset == "solar":
    pos0, vel0, masses, labels = presets.generate_solar_system(seed=42, with_moons=not args.no_moons)
elif args.preset == "solar_real":
    pos0, vel0, masses, labels = presets.generate_solar_system_real(with_moons=not args.no_moons)
elif args.preset == "solar_completo":
    pos0, vel0, masses, labels, names = presets.generate_solar_completo(with_moons=not args.no_moons, cache_path=args.cache, n_asteroids=args.asteroids, asteroids_cache=args.asteroids_cache)
else:
    pos0, vel0, masses, labels = presets.generate_spiral_galaxy(N_disk=int(args.N*0.7), N_bulge=args.N-int(args.N*0.7), seed=42)
    names = None

compute_forces, _, tag = _get_backends(args.backend)
pos = pos0.copy(); vel = vel0.copy()
accel = compute_forces(pos, masses, 1.0, args.eps)
N = len(masses)
print(f"N={N} [{args.backend}]")
if names is not None:
    print(f"Corpos: {', '.join(str(n) for n in list(names)[:9])} ... (+{N-9} luas/anões)")

from vispy import scene, app
from PIL import Image, ImageFilter

canvas = scene.SceneCanvas(keys='interactive', show=True, size=(900,700), bgcolor="black")
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 12
lim = np.percentile(np.abs(pos), 99)*1.3
lim = max(lim, 3.0)
view.camera.center = (0,0,0)
if args.topdown:
    # olhando de cima da eclíptica: forma do cinturão sem paralaxe de rotação
    try:
        view.camera.elevation = 90.0
        view.camera.azimuth = 0.0
    except Exception as e:
        print(f"[aviso] topdown indisponível: {e}")
# follow: ancóra câmera no corpo até Shift+drag desancorar
follow_idx = None
follow_enabled = False
if args.follow:
    # mapeia nome -> índice (usa nomes reais do preset quando houver) ou índice numérico
    try:
        # nomes reais (solar_completo) ou lista fixa (solar/solar_real)
        solar_names_all = list(names) if names is not None else ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
                           "Moon","Phobos","Deimos","Io","Europa","Ganymede","Callisto","Amalthea","Himalia","Elara","Pasiphae",
                           "Mimas","Tethys","Dione","Rhea","Iapetus","Titan","Enceladus","Miranda","Ariel","Umbriel","Oberon","Titania","Triton","Nereid",
                           "Ceres","Pluto","Haumea","Makemake","Eris"]
        if args.follow.lower() in [str(n).lower() for n in solar_names_all]:
            follow_idx = [str(n).lower() for n in solar_names_all].index(args.follow.lower())
            if follow_idx >= N:
                follow_idx = None
        else:
            # tenta por índice/label
            try:
                follow_idx = int(args.follow)
                if follow_idx < 0 or follow_idx >= N:
                    follow_idx = None
            except:
                follow_idx = None
    except Exception:
        follow_idx = None
    if follow_idx is not None:
        follow_enabled = True
        view.camera.center = tuple(pos[follow_idx].astype(float))
        print(f"Follow ancorado em {args.follow} idx {follow_idx} — Shift+drag desancora, <-/-> troca de corpo")
        def _on_mouse_press(ev):
            global follow_enabled
            # Shift pressionado durante drag desancora
            if ev.modifiers and 'Shift' in ev.modifiers:
                if follow_enabled:
                    follow_enabled = False
                    print("Follow desancorado (Shift+drag)")
        canvas.events.mouse_press.connect(_on_mouse_press)

# ângulos iniciais para --camera-orbit (lidos após topdown/follow configurarem a câmera)
try:
    _cam_az0 = float(view.camera.azimuth)
    _cam_el0 = float(view.camera.elevation)
except Exception:
    _cam_az0, _cam_el0 = 30.0, 30.0

# setas <-/-> trocam a âncora entre os corpos maiores (Sol, planetas, anões)
try:
    follow_cycle = [i for i in range(N) if labels is None or int(labels[i]) != 9]
except Exception:
    follow_cycle = list(range(N))
follow_pos = 0
if follow_idx in follow_cycle:
    follow_pos = follow_cycle.index(follow_idx)
# luas por corpo (para ↑/↓): cada lua vai para o corpo não-lua mais próximo;
# lista em ordem de distância orbital. Calculado 1x no startup.
moon_parent = {}
parent_moons = {}
try:
    _bodies = [i for i in range(N) if labels is None or int(labels[i]) != 9]
    _midx = [i for i in range(N) if labels is not None and int(labels[i]) == 9]
    if _midx and _bodies:
        _d = np.linalg.norm(pos[_midx][:, None, :] - pos[_bodies][None, :, :], axis=2)
        for _m, _k in zip(_midx, np.argmin(_d, axis=1)):
            _p = _bodies[int(_k)]
            moon_parent[int(_m)] = _p
            parent_moons.setdefault(_p, []).append(int(_m))
    for _p, _lst in parent_moons.items():
        _lst.sort(key=lambda i: float(np.linalg.norm(pos[i] - pos[_p])))
except Exception:
    moon_parent = {}
    parent_moons = {}
def _on_key_press(ev):
    global follow_idx, follow_enabled, follow_pos
    if ev.key in ("Left", "Right") and follow_cycle:
        step_dir = -1 if ev.key == "Left" else 1
        follow_pos = (follow_pos + step_dir) % len(follow_cycle)
        follow_idx = follow_cycle[follow_pos]
        follow_enabled = True
        try:
            view.camera.center = tuple(pos[follow_idx].astype(float))
        except Exception:
            pass
        try:
            nm = str(list(names)[follow_idx]) if names is not None else str(follow_idx)
        except Exception:
            nm = str(follow_idx)
        print(f"Follow: {nm} (idx {follow_idx})")
        canvas.update()
    elif ev.key in ("Up", "Down") and parent_moons:
        # planeta atual: se a âncora é lua, usa o pai dela; senão a própria âncora
        cur = int(follow_idx) if follow_idx is not None else None
        par = moon_parent.get(cur, cur)
        lst = parent_moons.get(par, [])
        if not lst:
            print("Corpo atual sem luas registradas (ancore um planeta/lua primeiro)")
            return
        try:
            mi = lst.index(cur)
        except ValueError:
            mi = -1 if ev.key == "Up" else 0
        mi = (mi + (1 if ev.key == "Up" else -1)) % len(lst)
        follow_idx = lst[mi]
        follow_enabled = True
        try:
            view.camera.center = tuple(pos[follow_idx].astype(float))
        except Exception:
            pass
        try:
            nm = str(list(names)[follow_idx]) if names is not None else str(follow_idx)
        except Exception:
            nm = str(follow_idx)
        print(f"Follow: {nm} (idx {follow_idx}, lua {mi + 1}/{len(lst)})")
        canvas.update()
canvas.events.key_press.connect(_on_key_press)

scatter = scene.visuals.Markers(scaling='fixed', antialias=0)
if labels is not None:
    uniq=np.unique(labels)
    tab10=np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189]],dtype=float)/255
    colors=np.array([tab10[int(np.where(uniq==l)[0][0])%5] for l in labels], dtype=np.float32)
    colors=np.column_stack([colors, np.full(N,0.95)])
else:
    colors=np.ones((N,4), dtype=np.float32)
m_med = np.median(masses)
if not np.isfinite(m_med) or m_med <= 0:
    _pos_m = masses[masses > 0]
    m_med = float(np.median(_pos_m)) if len(_pos_m) else 1.0  # 50k asteroides massa 0 zeravam a mediana
with np.errstate(divide="ignore", invalid="ignore"):
    sizes = (args.size * (masses / m_med) ** 0.33).astype(np.float32)
sizes = np.clip(np.nan_to_num(sizes, nan=1.0, posinf=args.size * 3.0), args.size*0.6, args.size*3.0)
# solar: cores realistas e tamanhos fixos para identificar — vale para solar, solar_real e solar_completo
if args.preset in ("solar", "solar_real", "solar_completo") and labels is not None:
    solar_colors = {0:[1,0.9,0.1], 1:[0.6,0.6,0.6], 2:[0.9,0.6,0.2], 3:[0.2,0.5,1.0], 4:[1,0.25,0.15], 5:[0.9,0.65,0.15], 6:[0.93,0.85,0.4], 7:[0.45,0.85,0.95], 8:[0.25,0.35,0.9], 9:[0.75,0.75,0.75], 10:[0.6,0.9,0.6], 11:[0.55,0.45,0.35]}
    for i,l in enumerate(labels):
        if int(l) in solar_colors:
            colors[i,:3] = solar_colors[int(l)]
    solar_sizes = {0:14, 1:3.5, 2:4.2, 3:4.8, 4:3.8, 5:9, 6:8, 7:6, 8:6, 9:2.2, 10:2.5, 11:1.5}
    for i,l in enumerate(labels):
        sizes[i] = solar_sizes.get(int(l), args.size)
scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
view.add(scatter)
# labels nomeados (solar) — usa nomes reais do preset quando houver
# --labels nomeia TODOS os corpos presentes (limite com --moons, ex: --moons 50)
# Offset do label em relação ao corpo (args.label_dz; 0 = colado, anchor bottom joga o texto p/ cima do ponto)
LABEL_DZ = args.label_dz
if args.labels and args.preset in ("solar", "solar_real", "solar_completo"):
    if names is not None:
        solar_names = [str(n) for n in list(names)[:N]]
    else:
        solar_names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
                       "Moon","Phobos","Deimos","Io","Europa","Ganymede","Callisto","Amalthea","Himalia","Elara","Pasiphae",
                       "Mimas","Tethys","Dione","Rhea","Iapetus","Titan","Enceladus","Miranda","Ariel","Umbriel","Oberon","Titania","Triton","Nereid",
                       "Ceres","Pluto","Haumea","Makemake","Eris"]
        solar_names = solar_names[:N]
    keep = list(enumerate(solar_names))
    if args.moons is not None and labels is not None:
        # --moons limita só os labels das luas (primeiras do cache = maiores);
        # a simulação sempre tem todas as luas genuínas
        _keep2 = []
        _seen = 0
        for i, nm in keep:
            if int(labels[i]) != 9:
                _keep2.append((i, nm))
            elif _seen < args.moons:
                _keep2.append((i, nm))
                _seen += 1
        keep = _keep2
    # 1 único Text p/ todos os labels (1 draw call, 1 atlas): pos é array (K,3) atualizado por frame
    label_idx = np.array([i for i, _ in keep], dtype=np.int64)
    label_names = [str(name) for _, name in keep]
    label_single = None
    label_texts = []
    if len(keep):
        try:
            label_single = scene.visuals.Text(text=label_names, color='white', font_size=args.label_size,
                pos=np.ascontiguousarray(pos[label_idx].astype(np.float32)) + np.array([0,0,LABEL_DZ],dtype=np.float32),
                parent=view.scene, anchor_x='center', anchor_y='bottom')
        except Exception as e:
            print(f"[aviso] Text único falhou ({e}), usando 1 Text por corpo")
            label_single = None
    if label_single is None and len(keep):
        for i, name in keep:
            # fallback: 1 Text por corpo (mais draw calls, mesmo resultado)
            txt = scene.visuals.Text(name, color='white', font_size=args.label_size, pos=pos[i].astype(np.float32) + np.array([0,0,LABEL_DZ],dtype=np.float32), parent=view.scene, anchor_x='center', anchor_y='bottom')
            label_texts.append((i, txt))
else:
    label_texts = []
    label_single = None
    label_idx = np.zeros(0, dtype=np.int64)

# eixos XYZ fixos no mundo (referencial inercial; parados enquanto a câmera orbita)
if args.axes:
    try:
        _ax_len = float(args.axes_len) if float(args.axes_len) > 0 else float(lim)
        _ax_cols = {"X": (1, 0.25, 0.25, 1), "Y": (0.25, 1, 0.25, 1), "Z": (0.35, 0.6, 1, 1)}
        for _ax, _vec in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
            _pts = np.array([[0, 0, 0], [v * _ax_len for v in _vec]], dtype=np.float32)
            scene.visuals.Line(pos=_pts, color=_ax_cols[_ax], width=2,
                               parent=view.scene, method='gl')
            _tip = np.array([[v * _ax_len * 0.90 for v in _vec]], dtype=np.float32)
            scene.visuals.Text(_ax, color='white', font_size=max(float(args.label_size) * 3.0, 0.05),
                               pos=_tip, parent=view.scene,
                               anchor_x='center', anchor_y='middle')
    except Exception as e:
        print(f"[aviso] eixos indisponíveis: {e}")

# raio Sol->Terra (direção da luz)
sun_ray = None
sun_idx = earth_idx = None
if args.sunray and args.preset in ("solar", "solar_real", "solar_completo") and labels is not None:
    try:
        sun_idx = int(np.where(labels == 0)[0][0])
        earth_idx = int(np.where(labels == 3)[0][0])
        sun_ray = scene.visuals.Line(pos=np.stack([pos[sun_idx], pos[earth_idx]]).astype(np.float32),
                                      color=[1, 0.9, 0.2, 0.9], width=2, parent=view.scene, method='gl')
    except Exception as e:
        print(f"[aviso] sunray indisponível: {e}")
        sun_ray = None

# bloom image atrás (cobre tela toda, ADD)
bloom_img = scene.visuals.Image(parent=view.scene, method='subdivide')
bloom_img.set_gl_state('translucent', blend=True, blend_func=('src_alpha','one'))
bloom_img.order = -10
scatter.order = 0
# posiciona bloom para cobrir view (world coords -lim a lim)
# Image em view.scene precisa estar em world coords, então cria uma imagem que cobre o view
# Para simplificar, usa Image em canvas.scene (screen coords) — mas view.scene é 3D, então usa canvas.scene
bloom_img2 = scene.visuals.Image(parent=canvas.scene, method='subdivide')
bloom_img2.set_gl_state('translucent', blend=True, blend_func=('src_alpha','one'))
# esconde o primeiro bloom_img (view.scene) e usa canvas.scene
bloom_img.visible = False

text = scene.visuals.Text("", color="white", font_size=9, pos=(10,20), parent=canvas.scene)
step=0; t_last=time.perf_counter(); fps_avg=60; t_compute_avg=0.5
accel_global=accel
# gravação de vídeo: pipe de frames RGBA para o ffmpeg
rec_proc = None
rec_frames = 0
if args.record:
    import subprocess as _sp
    try:
        import imageio_ffmpeg as _iiff
        _ff = _iiff.get_ffmpeg_exe()
    except Exception:
        _ff = "ffmpeg"
    import pathlib as _pl
    _pl.Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    _W, _H = int(canvas.size[0]), int(canvas.size[1])
    rec_proc = _sp.Popen([_ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
                          "-s", f"{_W}x{_H}", "-r", str(float(args.record_fps)),
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "20", "-movflags", "+faststart", args.record],
                         stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)


def _rec_finalize():
    global rec_proc
    if rec_proc is not None:
        try:
            rec_proc.stdin.close()
            rec_proc.wait(timeout=120)
            print(f"[record] vídeo salvo ({rec_frames} frames)")
        except Exception as e:
            print(f"[record] aviso ao finalizar: {e}")
        rec_proc = None
# trails: pontos atrás — número ou "inf" para nunca apagar
if args.trails is None:
    trail_len = 0
else:
    v = str(args.trails).lower()
    if v in ("inf", "infinite", "true", "yes"):
        trail_len = args.steps  # inf: nunca apaga
    else:
        try:
            trail_len = int(float(v))
            if trail_len < 0:
                trail_len = 0
        except:
            trail_len = 40
if trail_len:
    # se trail_len == steps -> nunca apaga; senão circular com trail_len pontos
    max_hist = trail_len if trail_len <= 20000 else 20000
    trail_hist = np.zeros((max_hist, N, 3), dtype=np.float32)
    trail_hist[0] = pos.astype(np.float32)
    trail_idx = 1
    max_trails = min(N, 30)
    trail_markers = scene.visuals.Markers(scaling='fixed', antialias=0, parent=view.scene)
    trail_markers.set_gl_state('translucent', blend=True, depth_test=False)
    trail_max_hist = max_hist
else:
    trail_hist = None  # type: ignore
    trail_max_hist = 0

def update(ev):
    global pos, vel, accel_global, step, t_last, fps_avg, t_compute_avg, trail_hist, trail_idx, trail_max_hist, follow_idx, follow_enabled, rec_frames
    if step >= args.steps:
        _rec_finalize()
        if args.hold:
            canvas.title = f"congelado step {step}/{args.steps} (hold)"
            canvas.update()
            return
        app.quit()
        return
    # speed agora é só reprodução (frames), não dt — física sempre dt fixo para não perder precisão
    dt_eff = args.dt
    if args.adaptive and args.preset in ("solar", "solar_real", "solar_completo"):
        max_a = float(np.max(np.linalg.norm(accel_global, axis=1)))
        eps_a = args.eps if args.eps > 1e-12 else 0.005
        dt_eff = float(np.clip(0.1 * np.sqrt(eps_a / (max_a + 1e-12)), args.dt*0.05, args.dt))
    # reproduz speed passos de física por frame sem alterar dt (mantém precisão)
    t0 = time.perf_counter()
    for _ in range(max(1, int(args.speed))):
        if args.integrator == "euler":
            accel = compute_forces(pos, masses, 1.0, args.eps)
            vel += accel * dt_eff
            pos += vel * dt_eff
            accel_global = accel
        else:  # leapfrog
            vel_half = vel + accel_global * (dt_eff * 0.5)
            pos[:] = pos + vel_half * dt_eff
            accel_new = compute_forces(pos, masses, 1.0, args.eps)
            vel[:] = vel_half + accel_new * (dt_eff * 0.5)
            accel_global[:] = accel_new
        if args.adaptive and args.preset in ("solar", "solar_real", "solar_completo"):
            # recalcula dt adaptativo a cada sub-step se necessário
            max_a = float(np.max(np.linalg.norm(accel_global, axis=1)))
            eps_a = args.eps if args.eps > 1e-12 else 0.005
            dt_eff = float(np.clip(0.1 * np.sqrt(eps_a / (max_a + 1e-12)), args.dt*0.05, args.dt))
    step += 1
    if trail_len:
        # inf (trail_len == steps) -> nunca apaga, só cresce até max_hist
        # número (ex: 40) -> circular com trail_len pontos
        is_inf = (trail_len == args.steps)
        if is_inf:
            if trail_idx < trail_max_hist:
                trail_hist[trail_idx] = pos.astype(np.float32)
                n_hist = trail_idx + 1
                hist = trail_hist[:n_hist]
            else:
                hist = trail_hist
                n_hist = trail_max_hist
        else:
            # circular com trail_len pontos
            trail_hist[trail_idx % trail_len] = pos.astype(np.float32)
            n_hist = min(trail_idx + 1, trail_len)
            if trail_idx < trail_len:
                hist = trail_hist[:n_hist]
            else:
                start = (trail_idx + 1) % trail_len
                hist = np.concatenate([trail_hist[start:], trail_hist[:start]], axis=0)
        try:
            pts = hist[:, :max_trails, :].reshape(-1, 3)
            base_colors = colors[:max_trails]
            alphas = np.linspace(0.15, 0.85, n_hist, dtype=np.float32)[:, None]
            cols = np.repeat(base_colors[None, :, :], n_hist, axis=0).reshape(-1, 4)
            cols[:, 3] *= np.repeat(alphas, max_trails, axis=0).flatten()
            sizes_trail = np.full(len(pts), 2.2, dtype=np.float32)
            trail_markers.set_data(pts.astype(np.float32), face_color=cols.astype(np.float32), size=sizes_trail, edge_width=0)
        except Exception:
            pass
        trail_idx += 1
    t_compute=(time.perf_counter()-t0)*1000
    t_compute_avg=0.9*t_compute_avg+0.1*t_compute if 't_compute_avg' in globals() else t_compute
    scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
    if sun_ray is not None:
        try:
            sun_ray.set_data(np.stack([pos[sun_idx], pos[earth_idx]]).astype(np.float32))
            if follow_enabled and follow_idx == earth_idx:
                d = pos[sun_idx] - pos[earth_idx]
                n = np.linalg.norm(d)
                if n > 1e-12:
                    view.camera.up = tuple((d / n).astype(float))
        except Exception:
            pass
    if follow_enabled and follow_idx is not None:
        try:
            view.camera.center = tuple(pos[follow_idx].astype(float))
        except Exception:
            pass
    if 'dt_eff' in locals() and args.adaptive:
        # mostra dt adaptativo no título occasionalmente
        if step % 30 == 0:
            canvas.title = f"step {step}/{args.steps}  dt {dt_eff:.5f}  compute {t_compute:.1f}ms  FPS {fps_avg:.0f} adaptive"
    if label_single is not None:
        try:
            label_single.pos = np.ascontiguousarray(pos[label_idx].astype(np.float32)) + np.array([0,0,LABEL_DZ], dtype=np.float32)
        except Exception:
            pass
    elif label_texts:
        for i, txt in label_texts:
            try:
                txt.pos = pos[i].astype(np.float32) + np.array([0,0,LABEL_DZ], dtype=np.float32)
            except Exception:
                pass
    if args.debug_labels and step == 30:
        try:
            dbg_i = int(label_idx[0]) if len(label_idx) else 0
            dbg_lp = np.asarray(label_single.pos)[0] if label_single is not None else None
            print(f"[debug-labels] body[{dbg_i}] world={pos[dbg_i]} label_world={dbg_lp} cam_dist={view.camera.distance}")
            img = canvas.render(alpha=True)
            from PIL import Image as _PILImage
            _PILImage.fromarray(img).save("debug_labels.png")
            print("[debug-labels] salvo debug_labels.png (confira onde o texto está vs o corpo)")
        except Exception as e:
            print(f"[debug-labels] falhou: {e}")
    if args.glow and step % 3 == 0:
        try:
            img = canvas.render(alpha=True)
            from PIL import Image, ImageFilter
            pil = Image.fromarray(img)
            W2, H2 = pil.size
            small = pil.resize((W2//2, H2//2), Image.BILINEAR)
            small = small.filter(ImageFilter.GaussianBlur(radius=0.001))
            bloom = small.resize((W2, H2), Image.BILINEAR)
            bloom_np = (np.array(bloom).astype(np.float32) * 0.95).astype(np.uint8)
            bloom_img2.set_data(bloom_np)
            bloom_img2.visible = True
        except Exception:
            pass
    elif not args.glow:
        bloom_img2.visible = False
    if args.camera_orbit:
        try:
            prog = min(max(step / max(int(args.steps), 1), 0.0), 1.0)
            view.camera.azimuth = _cam_az0 + float(args.orbit_az) * prog
            ph = (float(args.orbit_el_cycles) * prog) % 1.0
            tri = 1.0 - abs(2.0 * ph - 1.0)  # 0→1→0 por ciclo
            view.camera.elevation = _cam_el0 * (1.0 - tri)
        except Exception:
            pass
    canvas.update()
    if rec_proc is not None and step % max(1, int(args.record_every)) == 0:
        try:
            frame = canvas.render(alpha=True)
            rec_proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            rec_frames += 1
        except (BrokenPipeError, ValueError):
            _rec_finalize()
        except Exception:
            pass
    # HUD
    dt_total=time.perf_counter()-t_last
    t_last=time.perf_counter()
    fps_avg=0.9*fps_avg+0.1*(1/max(dt_total,1e-6))
    if getattr(args, "_until_epoch", None) and step % 30 == 0:
        try:
            import datetime as _dtm
            _d = _dtm.datetime.fromisoformat(args._until_epoch) + _dtm.timedelta(days=step * max(1, int(args.speed)) * dt_eff * args._until_T * 365.25)
            canvas.title = f"{_d.strftime('%d-%b-%Y')}  step {step}/{args.steps}  compute {t_compute:.1f}ms  FPS {fps_avg:.0f}"
        except Exception:
            canvas.title = f"step {step}/{args.steps}  compute {t_compute:.1f}ms  FPS {fps_avg:.0f}"
    else:
        canvas.title = f"step {step}/{args.steps}  compute {t_compute:.1f}ms  FPS {fps_avg:.0f}"

def on_draw(event):
    pass  # HUD já atualizado em update

canvas.events.draw.connect(on_draw)
timer = app.Timer(interval=0.0, connect=update, start=True)

if __name__ == "__main__":
    app.run()
