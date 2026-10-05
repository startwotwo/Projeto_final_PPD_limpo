#!/usr/bin/env python
"""
Propaga o sistema solar de 2026-01-01 até 2026-09-28 e compara com o cache de hoje.
Valida a precisão da simulação: a Lua propagada bate com a Lua real de hoje?

Uso:
  .venv312\\Scripts\\python scripts/propagate_to_today.py [--steps 0] [--backend numba]
  --steps 0 = calcula steps p/ cair exatamente em 28/09 (padrão)
"""
import sys
import time
import argparse
import datetime as _dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import src.presets as presets
import src.presets_runner_fast as runner

EPOCH0 = "2026-01-01"
EPOCH1 = "2026-09-28"
CACHE0 = "data/horizons_cache.json"
CACHE1 = "data/horizons_cache_20260928.json"
G = 1.0
EPS = 1e-5
DT = 2e-05
KM_PER_UNIT = 10 * 149597870.7  # 1 unidade sim = 10 AU em km


M_SUN_SIM = 0.998  # mesma do preset (a escala de tempo depende dela!)


def _parse_when(s):
    """Aceita dia (2026-09-29), ISO com hora ('2026-09-26 16:49') ou com traços (2026-09-26-16-49)."""
    import re as _re
    s = _re.sub(r"^(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})$", r"\1 \2:\3", s.strip())
    return _dt.datetime.fromisoformat(s)


def sim_time_between(d0, d1):
    """Converte intervalo real em tempo de simulação (mesma escala do preset)."""
    import math
    days = (_parse_when(d1) - _parse_when(d0)).total_seconds() / 86400.0
    years = days / 365.25
    T = math.sqrt(G * M_SUN_SIM * 1000.0 / (4 * math.pi ** 2))
    return years / T, days


def parse_pos(cache_path, sid):
    """Posição (S=0.1) do bloco $$SOE, sem shift de COM."""
    import json, re
    bodies = json.loads(Path(cache_path).read_text(encoding="utf-8"))["bodies"]
    raw = bodies[str(sid)].get("raw", "")
    m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line or line.startswith("$$"):
            continue
        nums = re.findall(r'[-+]?\d+\.\d+E[+-]\d+', line)
        if len(nums) >= 3:
            return np.array([float(nums[0]) * 0.1, float(nums[1]) * 0.1, float(nums[2]) * 0.1])
    raise RuntimeError(f"sem VECTORS p/ {sid} em {cache_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="0 = calcula p/ cair exato no alvo")
    ap.add_argument("--backend", default="numba")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--dt", type=float, default=0.0, help="0 = 2e-05 ajustado p/ cair exato")
    ap.add_argument("--until", default=EPOCH1, help="data alvo (padrão 2026-09-28)")
    ap.add_argument("--cache1", default=CACHE1, help="cache da verdade no alvo")
    args = ap.parse_args()
    target_epoch = args.until
    target_cache = args.cache1

    t_sim, days = sim_time_between(EPOCH0, target_epoch)
    dt_base = args.dt or DT
    steps = args.steps or int(round(t_sim / dt_base))
    dt = t_sim / steps
    eps = args.eps
    print(f"{EPOCH0} -> {target_epoch}: {days} dias = {t_sim:.6f} t.sim | {steps} steps dt={dt:.2e}")

    pos0, vel0, masses, labels, names = presets.generate_solar_completo(G=G, eps=eps, cache_path=CACHE0)
    names = [str(n) for n in names]
    i_earth = names.index("Earth")
    i_moon = names.index("Moon")
    i_sun = names.index("Sun")
    print(f"IC: N={len(masses)}")

    t0 = time.perf_counter()
    res = runner.run_fast(pos0, vel0, masses, steps=steps, dt=dt, G=G, eps=eps,
                          backend=args.backend, save_trajectory=False,
                          energy_every=max(1, steps // 5), verbose=True, labels=labels)
    print(f"propagação: {time.perf_counter() - t0:.1f}s")
    e0 = res.energy_history[0][2]
    e1 = res.energy_history[-1][2]
    print(f"drift energia: {abs(e1 - e0) / abs(e0):.2e}")

    posf = res.pos
    # vetores relativos (invariantes ao shift de COM)
    d_moon_sim = posf[i_moon] - posf[i_earth]
    d_sun_sim = posf[i_sun] - posf[i_earth]

    # verdade de hoje (cache 28/09): Terra helio + Lua planetocêntrica
    r_earth_t = parse_pos(target_cache, 399)
    d_moon_t = parse_pos(target_cache, 301)
    d_sun_t = -r_earth_t  # Sol - Terra (Sol na origem helio)

    err_moon = float(np.linalg.norm(d_moon_sim - d_moon_t))
    err_sun = float(np.linalg.norm(d_sun_sim - d_sun_t))
    dist_moon = float(np.linalg.norm(d_moon_t))
    dist_sun = float(np.linalg.norm(d_sun_t))
    print(f"\nTerra->Lua  sim={np.linalg.norm(d_moon_sim):.6f}  hoje={dist_moon:.6f} (unid)")
    print(f"erro Lua:   {err_moon:.6f} unid = {err_moon * KM_PER_UNIT:,.0f} km "
          f"({err_moon / dist_moon * 100:.2f}% da distância Terra-Lua)")
    print(f"erro Terra (Sol->Terra): {err_sun:.6f} unid = {err_sun * KM_PER_UNIT:,.0f} km "
          f"({err_sun / dist_sun * 100:.3f}% de 1 AU)")

    def elong(d_moon, d_sun):
        c = float(np.dot(d_moon, d_sun) / (np.linalg.norm(d_moon) * np.linalg.norm(d_sun)))
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    el_sim, el_t = elong(d_moon_sim, d_sun_sim), elong(d_moon_t, d_sun_t)
    print(f"elongação:  sim={el_sim:.2f}°  hoje={el_t:.2f}°  (cheia=180°)")
    print(f"iluminada:  sim={(1 + np.cos(np.radians(180 - el_sim))) / 2 * 100:.1f}%  "
          f"hoje={(1 + np.cos(np.radians(180 - el_t))) / 2 * 100:.1f}%")
    print("\nveredito:", "[OK] propagação bate com hoje" if err_moon / dist_moon < 0.05 else "[FAIL] divergiu")


if __name__ == "__main__":
    main()
