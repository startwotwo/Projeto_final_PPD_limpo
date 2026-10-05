#!/usr/bin/env python
"""Teste JWST: propaga Sol+Terra+Lua+JWST Jan->Set/2026 e compara com cache de Set.

Responde: diverge por bug de condicao inicial ou por instabilidade real do halo?
"""
import sys
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import src.presets as P
import src.nbody_numba as nb

G = 1.0
M_SUN_SIM = 0.998
T_time = math.sqrt(G * M_SUN_SIM * 1000.0 / (4 * math.pi ** 2))
S = 0.1

pos_j, vel_j, masses_j, _, names_j = P.generate_solar_completo(
    cache_path='data/horizons_cache.json', with_moons=True)
pos_s, vel_s, masses_s, _, names_s = P.generate_solar_completo(
    cache_path='data/horizons_cache_20260928.json', with_moons=True)
names_j = list(names_j)
i_sun = names_j.index('Sun')
i_earth = names_j.index('Earth')
i_moon = names_j.index('Moon')
i_jw = [i for i, n in enumerate(names_j) if 'Webb' in n][0]
print('JWST:', names_j[i_jw])

r_jan = float(np.linalg.norm((pos_j[i_jw] - pos_j[i_earth])) / S)
r_sep_real = float(np.linalg.norm((pos_s[i_jw] - pos_s[i_earth])) / S)
print(f'r_geoc jan={r_jan:.5f} AU  set(real)={r_sep_real:.5f} AU')

eps, dt = 1e-6, 1e-5
idx = [i_sun, i_earth, i_moon, i_jw]
p = pos_j[idx].copy()
v = vel_j[idx].copy()
m = masses_j[idx].copy()
days = 270.0
t_sim = (days / 365.25) / T_time
steps = int(round(t_sim / dt))
print(f'{days:.0f} dias = {t_sim:.6f} t.sim, {steps} steps dt={dt:g}')
acc = nb.compute_forces_numba(p, m, G, eps)
traj = []
for s in range(steps):
    vh = v + acc * (dt * 0.5)
    p = p + vh * dt
    acc = nb.compute_forces_numba(p, m, G, eps)
    v = vh + acc * (dt * 0.5)
    if s % max(1, steps // 8) == 0:
        traj.append(round(float(np.linalg.norm(p[3] - p[1]) / S), 4))
r_fin = float(np.linalg.norm(p[3] - p[1]) / S)
print('r_geoc(AU) jan->set:', traj)
print(f'final sim={r_fin:.5f} AU  vs real={r_sep_real:.5f} AU')
