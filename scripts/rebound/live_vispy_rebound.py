#!/usr/bin/env python3
"""
VisPy ao vivo com REBOUND IAS15 — 400 corpos JPL VECTORS heliocêntrico direto
Usa sim.integrate() 15ª ordem adaptativo, não leapfrog, para luas próximas
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse, time, numpy as np
import src.presets as presets

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="solar_completo", choices=["solar_completo","solar","solar_real","galaxy","collision"])
parser.add_argument("--eps", type=float, default=0.0)
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--size", type=float, default=6.0)
parser.add_argument("--follow", type=str, default="Jupiter")
parser.add_argument("--trails", action="store_true")
parser.add_argument("--labels", action="store_true")
parser.add_argument("--glow", action="store_true")
args = parser.parse_args()

# Build REBOUND sim com VECTORS JPL 2026-01-01 heliocêntrico direto (mesma de sim_rebound_vectors.py)
from scripts.sim_rebound_vectors import build_simulation
sim = build_simulation(limit=None if args.preset=="solar_completo" else 20)
# sim já tem 430 corpos VECTORS JPL 2026-01-01 heliocêntrico, massas reais, IAS15
N = sim.N
print(f"REBOUND IAS15 N={N} dt={args.dt} eps={args.eps} follow={args.follow}")

# Extrai pos/massas/labels iniciais para VisPy
import json, pathlib
# labels para cores: 0 Sol, 1-8 planetas, 9 luas, 10 anões (mesmo de presets)
# Para solar_completo, labels são 0,1-8,9,10
# Vamos usar massas do sim para size
masses = np.array([p.m for p in sim.particles], dtype=np.float64)
# labels aproximados por massa/ordem
labels = np.zeros(N, dtype=np.int32)
labels[0]=0
if N>1:
    labels[1:9]=np.arange(1,9)
    labels[9:]=9
    if N>430-5:
        labels[-5:]=10

from vispy import scene, app
from PIL import Image, ImageFilter

canvas = scene.SceneCanvas(keys='interactive', show=True, size=(900,700), bgcolor="black")
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 12
view.camera.center = (0,0,0)

scatter = scene.visuals.Markers(scaling='fixed', antialias=0)
# cores solares
tab10=np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207]],dtype=float)/255
uniq=np.unique(labels)
colors=np.array([tab10[int(l)%10] for l in labels], dtype=np.float32)
colors=np.column_stack([colors, np.full(N,0.95)])
# solar cores realistas
solar_colors={0:[1,0.9,0.1], 1:[0.6,0.6,0.6], 2:[0.9,0.6,0.2], 3:[0.2,0.5,1.0], 4:[1,0.25,0.15], 5:[0.9,0.65,0.15], 6:[0.93,0.85,0.4], 7:[0.45,0.85,0.95], 8:[0.25,0.35,0.9], 9:[0.75,0.75,0.75], 10:[0.6,0.6,0.6]}
for i,l in enumerate(labels):
    if int(l) in solar_colors:
        colors[i,:3]=solar_colors[int(l)]
m_med=np.median(masses)
sizes=(args.size * (masses/max(m_med,1e-12))**0.33).astype(np.float32)
sizes=np.clip(sizes, args.size*0.6, args.size*3.0)
solar_sizes={0:14, 1:3.5, 2:4.2, 3:4.8, 4:3.8, 5:9, 6:8, 7:6, 8:6, 9:2.2, 10:3}
for i,l in enumerate(labels):
    sizes[i]=solar_sizes.get(int(l), args.size)
pos0=np.array([[p.x,p.y,p.z] for p in sim.particles], dtype=np.float32)
scatter.set_data(pos0*10, face_color=colors, size=sizes, edge_width=0)  # *10 desfaz escala /10
view.add(scatter)
scatter.order=0
bloom_img2=scene.visuals.Image(parent=canvas.scene, method='subdivide')
bloom_img2.set_gl_state('translucent', blend=True, blend_func=('src_alpha','one'))
bloom_img2.visible=False

# follow
solar_names=["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune","Moon","Phobos","Deimos","Io","Europa","Ganymede","Callisto","Amalthea","Himalia","Elara","Pasiphae","Mimas","Tethys","Dione","Rhea","Iapetus","Titan","Enceladus","Miranda","Ariel","Umbriel","Oberon","Titania","Triton","Nereid","Ceres","Pluto","Haumea","Makemake","Eris"]
follow_idx=None
follow_enabled=False
if args.follow:
    try:
        if args.follow.lower() in [n.lower() for n in solar_names]:
            follow_idx=[n.lower() for n in solar_names].index(args.follow.lower())
            if follow_idx>=N:
                follow_idx=None
        else:
            follow_idx=int(args.follow)
        if follow_idx is not None:
            follow_enabled=True
            print(f"Follow {args.follow} idx {follow_idx}")
            def _on_press(ev):
                global follow_enabled
                if ev.modifiers and 'Shift' in ev.modifiers:
                    follow_enabled=False
                    print("Follow off Shift+drag")
            canvas.events.mouse_press.connect(_on_press)
    except: pass

# labels
if args.labels:
    solar_names=solar_names[:N]
    label_texts=[]
    for i,name in enumerate(solar_names):
        txt=scene.visuals.Text(name, color='white', font_size=7, pos=pos0[i]*10 + np.array([0,0,0.02],dtype=np.float32), parent=view.scene, anchor_x='center', anchor_y='bottom')
        label_texts.append(txt)
else:
    label_texts=[]

# trails
trail_len=80 if args.trails else 0
if trail_len:
    trail_hist=np.zeros((trail_len, N, 3), dtype=np.float32)
    trail_hist[0]=pos0*10
    trail_idx=1
    max_trails=min(N,30)
    trail_lines=[]
    for i in range(max_trails):
        line=scene.visuals.Line(pos=np.zeros((1,3),dtype=np.float32), color=colors[i] if i < len(colors) else [0.5,0.5,0.5,0.6], width=1, parent=view.scene, method='gl')
        trail_lines.append(line)
else:
    trail_hist=None

step=0; t_last=time.perf_counter(); fps_avg=60
def update(ev):
    global step, t_last, fps_avg, trail_hist, trail_idx
    if step >= 10000:
        app.quit()
        return
    t0=time.perf_counter()
    # REBOUND IAS15 adaptativo
    sim.integrate(sim.t + args.dt)
    pos=np.array([[p.x,p.y,p.z] for p in sim.particles], dtype=np.float32)*10
    step+=1
    if trail_len:
        trail_hist[trail_idx % trail_len]=pos
        try:
            n_hist=min(trail_idx, trail_len)
            for i,line in enumerate(trail_lines):
                if n_hist<2:
                    continue
                if trail_idx < trail_len:
                    hist=trail_hist[:n_hist,i]
                else:
                    hist=np.concatenate([trail_hist[trail_idx % trail_len:,i], trail_hist[:trail_idx % trail_len,i]], axis=0)
                line.set_data(hist.astype(np.float32))
        except: pass
        trail_idx+=1
    t_compute=(time.perf_counter()-t0)*1000
    scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
    if follow_enabled and follow_idx is not None:
        try:
            view.camera.center=tuple(pos[follow_idx].astype(float))
        except: pass
    if label_texts:
        for i,txt in enumerate(label_texts):
            try:
                txt.pos=pos[i].astype(np.float32)+np.array([0,0,0.015],dtype=np.float32)
            except: pass
    if args.glow and step%3==0:
        try:
            img=canvas.render(alpha=True)
            pil=Image.fromarray(img)
            W2,H2=pil.size
            small=pil.resize((W2//2,H2//2), Image.BILINEAR)
            small=small.filter(ImageFilter.GaussianBlur(radius=2))
            bloom=small.resize((W2,H2), Image.BILINEAR)
            bloom_np=(np.array(bloom).astype(np.float32)*0.6).astype(np.uint8)
            bloom_img2.set_data(bloom_np)
            bloom_img2.visible=True
        except: pass
    elif not args.glow:
        bloom_img2.visible=False
    canvas.update()
    dt_total=time.perf_counter()-t_last
    t_last=time.perf_counter()
    fps_avg=0.9*fps_avg+0.1*(1/max(dt_total,1e-6))
    canvas.title=f"REBOUND IAS15 step {step} t={sim.t:.3f} yr FPS {fps_avg:.0f} N={N}"

canvas.events.draw.connect(lambda ev: None)
timer=app.Timer(interval=0.0, connect=update, start=True)
print("REBOUND IAS15 ao vivo — Shift+drag desancora follow")
app.run()
