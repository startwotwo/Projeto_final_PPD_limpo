#!/usr/bin/env python
"""
Imagens top-down (matplotlib) dos presets do sistema solar real.
Gera docs/preset_images/solar_real_initial.png e solar_completo_initial.png
+ tabela de nomes (idx, nome, label) para conferir label->nome.
Com --asteroids N, gera também solar_completo_asteroids_initial.png.
Uso: .venv312\\Scripts\\python generate_solar_images.py [--asteroids 5000]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.presets as presets

OUTPUT_DIR = Path(__file__).resolve().parent / "docs" / "preset_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOLAR_COLORS = {
    0: (1, 0.9, 0.1), 1: (0.6, 0.6, 0.6), 2: (0.9, 0.6, 0.2), 3: (0.2, 0.5, 1.0),
    4: (1, 0.25, 0.15), 5: (0.9, 0.65, 0.15), 6: (0.93, 0.85, 0.4),
    7: (0.45, 0.85, 0.95), 8: (0.25, 0.35, 0.9), 9: (0.75, 0.75, 0.75), 10: (0.6, 0.9, 0.6),
    11: (0.55, 0.45, 0.35),
}
SOLAR_SIZES = {0: 120, 1: 12, 2: 16, 3: 20, 4: 14, 5: 45, 6: 40, 7: 28, 8: 28, 9: 4, 10: 10, 11: 1.5}


def cache_epoch(cache=None):
    """Lê a época real do cache (não hardcoded)."""
    try:
        import json
        cp = Path(cache) if cache else Path(__file__).resolve().parent / "data" / "horizons_cache.json"
        return json.loads(cp.read_text(encoding="utf-8")).get("epoch", "?")
    except Exception:
        return "?"


def plot_solar(name, pos, labels, names, epoch=None, zoom=1.0):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    colors = np.array([SOLAR_COLORS.get(int(l), (0.8, 0.8, 0.8)) for l in labels])
    sizes = np.array([SOLAR_SIZES.get(int(l), 4) for l in labels], dtype=float)
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=sizes, alpha=0.9, edgecolors="none")
    # anota Sol + planetas + anões (luas ficam só como pontos p/ não poluir)
    for i, (l, nm) in enumerate(zip(labels, names)):
        if int(l) in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10):
            ax.text(pos[i, 0], pos[i, 1], f" {nm}", color="white", fontsize=8)
    ax.set_xlabel("X (AU/10)", color="white")
    ax.set_ylabel("Y (AU/10)", color="white")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")
    ax.set_title(f"{name} — N={len(labels)} (vista de cima, época {epoch})", color="white")
    lim = float(np.max(np.abs(pos[:, :2]))) * 1.1 / zoom
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout()
    out = OUTPUT_DIR / f"{name}_initial.png"
    plt.savefig(out, dpi=150, facecolor="black")
    plt.close()
    print(f"OK {out}")


def save_names_table(name, labels, names):
    out = OUTPUT_DIR / f"{name}_names.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {name}: idx | nome | label (0=Sol,1-8=planetas,9=lua,10=anao)\n")
        for i, (nm, l) in enumerate(zip(names, labels)):
            f.write(f"{i:4d} | {nm} | {int(l)}\n")
    print(f"OK {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asteroids", type=int, default=0, help="anexa N asteroides reais (gera solar_completo_asteroids_initial.png)")
    ap.add_argument("--asteroids-cache", default=None)
    ap.add_argument("--cache", default=None, help="cache Horizons (padrão data/horizons_cache.json)")
    ap.add_argument("--zoom", type=float, default=8.0, help="zoom da imagem com asteroides (padrão 8x)")
    ap.add_argument("--no-solar-real", action="store_true", help="não gera a imagem do solar_real (elementos fixos, não-API)")
    args = ap.parse_args()

    if not args.no_solar_real:
        pos, vel, masses, labels = presets.generate_solar_system_real(with_moons=True)
        names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
                 "Moon", "Phobos", "Deimos", "Io", "Europa", "Ganymede", "Callisto", "Titan", "Enceladus", "Triton",
                 "Ceres", "Pluto", "Haumea", "Makemake", "Eris"][:len(labels)]
        plot_solar("solar_real", pos, labels, names, epoch="J2000 (elementos aproximados)")
        save_names_table("solar_real", labels, names)

    ep = cache_epoch(args.cache)
    pos, vel, masses, labels, names = presets.generate_solar_completo(with_moons=True, cache_path=args.cache)
    plot_solar("solar_completo", pos, labels, [str(n) for n in names], epoch=f"{ep} (Horizons)")
    save_names_table("solar_completo", labels, [str(n) for n in names])
    print(f"\nN solar_completo = {len(labels)}")

    if args.asteroids:
        pos, vel, masses, labels, names = presets.generate_solar_completo(
            with_moons=True, cache_path=args.cache,
            n_asteroids=args.asteroids, asteroids_cache=args.asteroids_cache)
        plot_solar("solar_completo_asteroids", pos, labels, [str(n) for n in names],
                   epoch=f"{ep} (Horizons + {args.asteroids} asteroides MPCORB)")
        print(f"N solar_completo_asteroids = {len(labels)}")
        plot_solar("solar_completo_asteroids_zoom", pos, labels, [str(n) for n in names],
                   epoch=f"{ep} + {args.asteroids} MPCORB (zoom {args.zoom:g}x)",
                   zoom=args.zoom)


if __name__ == "__main__":
    main()
