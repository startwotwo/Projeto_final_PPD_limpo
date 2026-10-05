#!/usr/bin/env python
"""
Generate initial state visualizations for all presets.
Saves top-down view (XY plane) as PNG files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import src.presets as presets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "docs" / "preset_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRESETS_CONFIG = {
    "galaxy": {"func": "generate_spiral_galaxy", "kwargs": {"N_disk": 1400, "N_bulge": 600}},
    "collision": {"func": "generate_galaxy_collision", "kwargs": {}},
    "plummer": {"func": "generate_plummer_sphere", "kwargs": {"N": 1000}},
    "rotplummer": {"func": "generate_rotating_plummer", "kwargs": {"N": 800}},
    "ring": {"func": "generate_ring_galaxy", "kwargs": {"N_disk": 1400, "N_bulge": 600}},
    "triple": {"func": "generate_triple_merger", "kwargs": {"N_per_galaxy": 500}},
    "satellite": {"func": "generate_satellite_infall", "kwargs": {"N_host_disk": 1400, "N_host_bulge": 600, "N_sat": 200}},
    "cold": {"func": "generate_cold_collapse", "kwargs": {"N": 1000}},
    "kepler": {"func": "generate_kepler_binary", "kwargs": {}},
    "disk": {"func": "generate_kepler_disk", "kwargs": {"N": 500}},
    "forming": {"func": "generate_forming_galaxy", "kwargs": {"N": 1000}},
    "solar_real": {"func": "generate_solar_system_real", "kwargs": {"with_moons": True}},
    "solar_completo": {"func": "generate_solar_completo", "kwargs": {"with_moons": True}},
}

def plot_preset(name, pos, *rest):
    """Plot top-down view of preset (matplotlib, sem VisPy)."""
    # Unpack rest (could be (vel, masses, labels) or (vel, masses))
    labels = None
    masses = None
    if len(rest) >= 3:
        _, masses, labels = rest[0], rest[1], rest[2]
    elif len(rest) == 2:
        _, masses = rest
    elif len(rest) == 1:
        masses = rest[0] if rest[0] is not None else None
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    
    # Color by label if available
    if labels is not None:
        uniq = np.unique(labels)
        cmap = plt.cm.tab10
        colors = np.array([cmap(i % 10) for i in labels])
    else:
        # Color by mass
        if masses is not None:
            norm = (masses - masses.min()) / (masses.max() - masses.min() + 1e-9)
            colors = plt.cm.viridis(norm)
        else:
            colors = np.ones((len(pos), 4))
            colors[:, :3] = 0.8
    
    # Size by mass
    if masses is not None:
        m_norm = masses / np.median(masses)
        sizes = np.clip(3 * m_norm ** 0.33, 1, 20)
    else:
        sizes = 3
    
    # Plot XY (top-down)
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=sizes, alpha=0.8, edgecolors='none')
    
    ax.set_xlabel('X (AU/10)', color='white')
    ax.set_ylabel('Y (AU/10)', color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    ax.set_title(name.replace('_', ' ').title(), color='white', fontsize=14)
    
    # Auto-scale with margin
    margin = 1.2
    lim = np.max(np.abs(pos[:, :2])) * margin
    lim = max(lim, 0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_initial.png", dpi=150, facecolor='black')
    plt.close()
    print(f"Saved {name}_initial.png")

def main():
    print("Generating preset visualizations...")
    for name, config in PRESETS_CONFIG.items():
        try:
            func = getattr(presets, config["func"])
            result = func(**config["kwargs"])
            plot_preset(name, *result)
        except Exception as e:
            print(f"Failed {name}: {e}")
    print(f"\nAll images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()