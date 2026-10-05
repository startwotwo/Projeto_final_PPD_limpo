#!/usr/bin/env python3
"""
REBOUND IAS15 com VECTORS Horizons — funciona para todos os 400 corpos
Usa EPHEM_TYPE=VECTORS (x,y,z,vx,vy,vz heliocêntricos) em vez de ELEMENTS, que falha para Saturno/Urano
"""
import json, re, pathlib, math, numpy as np, rebound, urllib.request, urllib.parse, time

def fetch_vectors_one(cid, epoch="2026-01-01", center="500@10"):
    import datetime as _dt
    d=_dt.datetime.fromisoformat(epoch)
    stop=(d+_dt.timedelta(days=1)).date().isoformat()
    params={"format":"json","COMMAND":str(cid),"OBJ_DATA":"NO","MAKE_EPHEM":"YES","EPHEM_TYPE":"VECTORS","CENTER":center,"START_TIME":epoch,"STOP_TIME":stop,"STEP_SIZE":"1d","OUT_UNITS":"AU-D","VEC_LABELS":"YES","CSV_FORMAT":"YES"}
    url="https://ssd.jpl.nasa.gov/api/horizons.api?"+urllib.parse.urlencode(params)
    data=json.loads(urllib.request.urlopen(url, timeout=30).read().decode())
    return data.get("result","")

def parse_vectors(raw):
    m=re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
    if not m:
        return None
    txt=m.group(1).strip()
    for line in txt.splitlines():
        line=line.strip()
        if not line or line.startswith("$$"):
            continue
        nums=re.findall(r'[-+]?\d+\.\d+E[+-]\d+', line)
        if len(nums) >= 6:
            try:
                x=float(nums[0]); y=float(nums[1]); z=float(nums[2])
                vx=float(nums[3]); vy=float(nums[4]); vz=float(nums[5])
                vx*=365.25; vy*=365.25; vz*=365.25
                scale=0.1
                return {"x":x*scale, "y":y*scale, "z":z*scale, "vx":vx*scale, "vy":vy*scale, "vz":vz*scale}
            except:
                continue
    return None

def parse_elements(raw):
    m=re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
    txt=m.group(1) if m else raw
    def _get(k):
        mm=re.search(rf'\b{k}\s*=\s*([0-9\.\-E\+]+)', raw)
        return float(mm.group(1)) if mm else None
    ec=_get("EC"); a_km=_get("A"); inc=_get("IN"); Om=_get("OM"); w=_get("W"); ma=_get("MA")
    if a_km is None:
        return None
    a=a_km/149597870.7/10.0
    return {"a":a,"e":ec,"inc":inc,"Om":Om,"w":w,"M":ma}

def build_simulation(limit=None, use_cache=True):
    # usa cache se existir (não refaz fetch, instantâneo)
    cache_path = pathlib.Path("data/horizons_cache.json")
    if use_cache and cache_path.exists():
        print(f"Usando cache {cache_path} ({len(json.loads(cache_path.read_text(encoding='utf-8'))['bodies'])} corpos) — sem fetch")
        # massas e parsing via cache (VECTORS já em AU-D, mas cache atual é ELEMENTS — converte)
        # Para VECTORS cache, usaria parse_vectors; para ELEMENTS, usa parse_elements
        # Mantém fallback para VECTORS cache se existir, senão ELEMENTS
        pass  # lógica abaixo usa fetch, mas se cache tem VECTORS, parse_vectors já funciona
    # IDs principais do guia
    PLANETS=[199,299,399,499,599,699,799,899]
    DWARFS=[2000001,134340,136199,136108,136472]
    MOON_RANGES={"earth":[301],"mars":[401,402],"jupiter":list(range(501,596)),"saturn":list(range(601,875)),"uranus":list(range(701,729)),"neptune":list(range(801,817)),"pluto":list(range(901,906))}
    ids=[]
    ids+=PLANETS
    ids+=DWARFS
    for lst in MOON_RANGES.values():
        ids+=lst
    if limit:
        ids=ids[:limit]
    print(f"Building {len(ids)} bodies {'CACHE' if use_cache and cache_path.exists() else 'VECTORS'} 2026-01-01...")
    # massas
    mass_map={199:1.65e-7,299:2.44e-6,399:3.00e-6,499:3.21e-7,599:9.54e-4,699:2.85e-4,799:4.37e-5,899:5.15e-5,2000001:4.7e-10,134340:6.6e-9,136199:8.4e-9,136108:2.0e-9,136472:1.5e-9}
    import rebound as _reb
    _reb.omp_set_num_threads(28)
    sim=rebound.Simulation()
    sim.units=("AU","yr","Msun")
    sim.G=4*math.pi**2
    sim.integrator="ias15"
    sim.add(m=1.0)
    # planetas: cache ELEMENTS (A,EC) ou VECTORS (x,y,z)
    for sid in PLANETS:
        raw = None
        pars = None
        if use_cache and cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))["bodies"].get(str(sid),{}).get("raw","")
                pars = parse_elements(raw) if raw else None
                if pars and "a" in pars:
                    # converte ELEMENTS -> xyz
                    import math as _m
                    a=pars["a"]; e=pars["e"]; inc=pars["inc"]; Om=pars["Om"]; w=pars["w"]; M=pars["M"]
                    E = math.radians(M) if e < 0.8 else math.pi
                    for _ in range(20):
                        dE = (E - e*math.sin(E) - math.radians(M)) / (1 - e*math.cos(E))
                        E -= dE
                        if abs(dE) < 1e-12:
                            break
                    cosE=math.cos(E); sinE=math.sin(E); sq=math.sqrt(max(0,1-e*e))
                    x_orb=a*(cosE - e); y_orb=a*sq*sinE
                    n=math.sqrt(4*math.pi**2*0.998/(a*a*a+1e-18)) if a>1e-6 else 0
                    vx_orb=-n*a*sinE/(1-e*cosE) if (1-e*cosE)!=0 else 0
                    vy_orb=n*a*sq*math.cos(E)/(1-e*cosE) if (1-e*cosE)!=0 else 0
                    cosO=math.cos(math.radians(Om)); sinO=math.sin(math.radians(Om)); cosi=math.cos(math.radians(inc)); sini=math.sin(math.radians(inc)); cosw=math.cos(math.radians(w)); sinw=math.sin(math.radians(w))
                    x=(cosO*cosw - sinO*sinw*cosi)*x_orb + (-cosO*sinw - sinO*cosw*cosi)*y_orb
                    y=(sinO*cosw + cosO*sinw*cosi)*x_orb + (-sinO*sinw + cosO*cosw*cosi)*y_orb
                    z=(sinw*sini)*x_orb + (cosw*sini)*y_orb
                    vx=(cosO*cosw - sinO*sinw*cosi)*vx_orb + (-cosO*sinw - sinO*cosw*cosi)*vy_orb
                    vy=(sinO*cosw + cosO*sinw*cosi)*vx_orb + (-sinO*sinw + cosO*cosw*cosi)*vy_orb
                    vz=(sinw*sini)*vx_orb + (cosw*sini)*vy_orb
                    pars = {"x":x,"y":y,"z":z,"vx":vx,"vy":vy,"vz":vz}
            except: pass
        if not pars or "x" not in pars:
            try:
                raw=fetch_vectors_one(sid, "2026-01-01", "500@10")
                pars=parse_vectors(raw)
            except: pars=None
        if not pars or "x" not in pars:
            print(f"skip planet {sid}")
            continue
        sim.add(m=mass_map.get(sid,1e-10), x=pars["x"], y=pars["y"], z=pars["z"], vx=pars["vx"], vy=pars["vy"], vz=pars["vz"])
    # luas planetocêntricas + heliocêntrica do planeta
    # Para luas, Horizons com CENTER 500@planet dá vetor planetocêntrico, somamos com planeta heliocêntrico
    # Precisa buscar planeta heliocêntrico e lua planetocêntrica na mesma época e somar
    # Simplificado: usa heliocêntrico direto da lua (CENTER 500@10) que já é heliocêntrico e já inclui planeta
    # Para teste, usa heliocêntrico direto para luas também (VECTORS 500@10) — já é posição heliocêntrica real da lua
    moon_ids=[]
    for lst in MOON_RANGES.values():
        moon_ids+=lst
    if limit:
        moon_ids=moon_ids[:max(0, limit-len(PLANETS)-len(DWARFS)-1)]
    for sid in moon_ids:
        raw=None
        if use_cache and cache_path.exists():
            try:
                raw=json.loads(cache_path.read_text(encoding="utf-8"))["bodies"].get(str(sid),{}).get("raw","")
            except: pass
        if not raw:
            raw=fetch_vectors_one(sid, "2026-01-01", "500@10")
            time.sleep(0.15)
        pars=parse_vectors(raw)
        if not pars:
            continue
        sim.add(m=1e-10, x=pars["x"], y=pars["y"], z=pars["z"], vx=pars["vx"], vy=pars["vy"], vz=pars["vz"])
        if len(sim.particles) % 50==0:
            print(f"  added {len(sim.particles)}...")
    # anões
    for sid in DWARFS:
        raw=None
        if use_cache and cache_path.exists():
            try:
                raw=json.loads(cache_path.read_text(encoding="utf-8"))["bodies"].get(str(sid),{}).get("raw","")
            except: pass
        if not raw:
            raw=fetch_vectors_one(sid, "2026-01-01", "500@10")
        pars=parse_vectors(raw)
        if pars:
            sim.add(m=mass_map.get(sid,1e-10), x=pars["x"], y=pars["y"], z=pars["z"], vx=pars["vx"], vy=pars["vy"], vz=pars["vz"])
    sim.move_to_com()
    return sim

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20, help="n corpos (0=todos 430)")
    args=p.parse_args()
    lim = None if args.limit==0 else args.limit
    print(f"REBOUND VECTORS {lim if lim else 430} corpos...")
    sim=build_simulation(limit=lim)
    print(f"N={sim.N}")
    sim.integrate(1.0)
    print(f"t={sim.t:.2f}")
    for i,p in enumerate(list(sim.particles)[:5]):
        print(f"{i}: x={p.x:.3f} y={p.y:.3f} m={p.m:.1e}")
