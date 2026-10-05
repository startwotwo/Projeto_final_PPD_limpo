#!/usr/bin/env python
"""Grava um vídeo VisPy por preset (usa scripts/live_vispy.py --record).

Uso (da raiz do repo):
    .\\.venv312\\Scripts\\python scripts/record_all_presets.py [--only galaxy,collision] [--fps 30]

Saída: data/videos/<preset>.mp4 (janela abre/fecha sozinha por preset).
Para publicar no README, copiar para docs/videos/ (data/*.mp4 é gitignored).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# preset -> (N, steps, dt, backend, extras)
# Durações analíticas (G=1) onde há forma fechada; 2000 steps nos demais:
# - kepler: T=2π√(a³/GM), a=1, M=2 -> T≈4.443; 2 órbitas, dt=0.005 -> 1777
# - satellite: a=r_apo/(1+e)=4/1.7, M=1.15 -> T≈21.15; 1,1 órbitas -> 4652
# - cold: t_ff=√(3π/32Gρ), ρ=M/(4πR³/3), R=1.5, M=1 -> t_ff≈2.04;
#   colapso+bounce em ~3 t_ff, dt=0.002 -> 3061
# - forming: R=1, M=1 -> t_ff≈1.11; colapso+assentamento em ~5 t_ff -> 1111
PRESETS = {
    "galaxy":     (1500, 2000, 0.005, "gpu", ["--trails", "80"]),
    "collision":  (1500, 2000, 0.005, "gpu", ["--trails", "80"]),
    "plummer":    (1000, 2000, 0.005, "gpu", ["--trails", "80"]),
    "rotplummer": (1000, 2000, 0.005, "gpu", ["--trails", "80"]),
    "ring":       (1500, 2000, 0.005, "gpu", ["--trails", "80"]),
    "triple":     (1500, 2000, 0.005, "gpu", ["--trails", "80"]),
    "satellite":  (1000, 4652, 0.005, "gpu", ["--trails", "80"]),
    "cold":       (1000, 3061, 0.002, "gpu", ["--trails", "80"]),
    "kepler":     (0, 1777, 0.005, "gpu", ["--trails", "200"]),
    "disk":       (1000, 2000, 0.005, "gpu", ["--trails", "80"]),
    "forming":    (1000, 1111, 0.005, "gpu", ["--trails", "80"]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--only", default=None, help="subset separado por vírgula")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--outdir", default="data/videos")
args = ap.parse_args()

only = None
if args.only:
    only = [p.strip() for p in args.only.split(",") if p.strip()]

outdir = ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
ok, fail = [], []
for name, (N, steps, dt, backend, extras) in PRESETS.items():
    if only and name not in only:
        continue
    out = outdir / f"{name}.mp4"
    cmd = [PY, "scripts/live_vispy.py", "--preset", name, "--backend", backend,
           "--steps", str(steps), "--record", str(out),
           "--record-fps", str(args.fps), "--camera-orbit", "--axes"]
    if N:
        cmd += ["--N", str(N)]
    if dt:
        cmd += ["--dt", str(dt)]
    cmd += extras
    print(f"\n=== {name} -> {out} ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    (ok if r.returncode == 0 else fail).append(name)
print(f"\nOK ({len(ok)}): {', '.join(ok)}")
if fail:
    print(f"FALHARAM ({len(fail)}): {', '.join(fail)}")
