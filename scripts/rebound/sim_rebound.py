#!/usr/bin/env python3
"""
REBOUND IAS15 com JPL Horizons real — 8 planetas + Sol (388 corpos no cache, mas demo com 9 para teste rápido)
IAS15 = 15â ordém adaptativo, estável para luas próximas
"""
import json, re, math, numpy as np, rebound, pathlib

def parse_elements(raw: str):
    m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
    txt = m.group(1) if m else raw
    def _get(k):
        mm = re.search(rf'\b{k}\s*=\s*([0-9\.\-E\+]+)', raw)
        return float(mm.group(1)) if mm else None
    ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
    Om = _get("OM"); w = _get("W"); ma = _get("MA")
    if a_km is None:
        return None
    a = a_km / 149597870.7 / 10.0
    return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}

def M_to_f(e, M_deg):
    M = math.radians(M_deg)
    E = M if e < 0.8 else math.pi
    for _ in range(30):
        dE = (E - e*math.sin(E) - M) / (1 - e*math.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    return 2 * math.atan(math.sqrt((1+e)/(1-e)) * math.tan(E/2))

def build_simulation(limit=None):
    cache = json.loads(pathlib.Path("data/horizons_cache.json").read_text(encoding="utf-8"))
    bodies = cache["bodies"]
    mass_map = {199:1.65e-7,299:2.44e-6,399:3.00e-6,499:3.21e-7,599:9.54e-4,699:2.85e-4,799:4.37e-5,899:5.15e-5, 1:1.0, 134340:6.6e-9,136199:8.4e-9,136108:2.0e-9,136472:1.5e-9}
    sim = rebound.Simulation()
    sim.units = ("AU","yr","Msun")
    sim.G = 4*math.pi**2
    sim.integrator = "ias15"
    sim.add(m=1.0)  # Sol
    ids = [199,299,399,499,599,699,799,899]
    if limit:
        ids = ids[:limit]
    for sid in ids:
        raw = bodies.get(str(sid),{}).get("raw","")
        pars = parse_elements(raw)
        if not pars:
            print(f"skip {sid} sem ELEMENTS")
            continue
        f = M_to_f(pars["e"], pars["M"])
        try:
            sim.add(m=mass_map.get(sid,1e-10), a=pars["a"], e=pars["e"], inc=math.radians(pars["inc"]), Omega=math.radians(pars["Om"]), omega=math.radians(pars["w"]), f=f)
        except Exception as e:
            print(f"fail {sid}: {e}")
    sim.move_to_com()
    return sim

if __name__ == "__main__":
    print("Building REBOUND IAS15...")
    sim = build_simulation()
    print(f"N={sim.N} particles")
    print("Integrating 10 yr...")
    sim.integrate(10.0)
    print(f"done t={sim.t:.2f} yr")
    for i,p in enumerate(sim.particles):
        print(f"{i}: m={p.m:.1e} x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}")
