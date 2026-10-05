#!/usr/bin/env python3
"""
Grava REBOUND IAS15 offline para solar_completo 430 corpos -> data/rebound_430.npz
Uso: .\.venv312\Scripts\python scripts/rebound/record_rebound.py --years 10 --dt 0.01 --out data/rebound_430.npz
Depois: python scripts/live_vispy.py --preset solar_completo --backend rebound (lê npz)
Ou: python -c "import numpy as np; d=np.load('data/rebound_430.npz'); print(d['traj'].shape)"
"""
import argparse, pathlib, sys, json, re, math, numpy as np, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import rebound

def build_rebound_sim_from_cache(cache_path="data/horizons_cache.json", limit=None):
    cache=json.loads(pathlib.Path(cache_path).read_text(encoding="utf-8"))
    bodies=cache["bodies"]
    mass_map={199:1.65e-7,299:2.44e-6,399:3.00e-6,499:3.21e-7,599:9.54e-4,699:2.85e-4,799:4.37e-5,899:5.15e-5,2000001:4.7e-10,134340:6.6e-9,136199:8.4e-9,136108:2.0e-9,136472:1.5e-9}
    # IDs principais
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
    print(f"Building REBOUND {len(ids)} corpos VECTORS...")
    sim=rebound.Simulation()
    sim.units=("AU","yr","Msun")
    sim.G=4*math.pi**2
    sim.integrator="ias15"
    try:
        import rebound as _reb
        _reb.omp_set_num_threads(28)
    except: pass
    sim.add(m=1.0)
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
            if len(nums)>=6:
                try:
                    x=float(nums[0]); y=float(nums[1]); z=float(nums[2]); vx=float(nums[3]); vy=float(nums[4]); vz=float(nums[5])
                    vx*=365.25; vy*=365.25; vz*=365.25
                    scale=0.1
                    return {"x":x*scale,"y":y*scale,"z":z*scale,"vx":vx*scale,"vy":vy*scale,"vz":vz*scale}
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
    def M_to_f(e, M_deg):
        M=math.radians(M_deg)
        E=M if e<0.8 else math.pi
        for _ in range(30):
            dE=(E - e*math.sin(E) - M)/(1 - e*math.cos(E))
            E-=dE
            if abs(dE)<1e-14:
                break
        return 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E/2))
    # planetas e anões (ELEMENTS)
    for sid in PLANETS+DWARFS:
        raw=bodies.get(str(sid),{}).get("raw","")
        pars=None
        # tenta ELEMENTS primeiro (cache atual)
        parsE=parse_elements(raw) if raw else None
        if parsE and "a" in parsE and parsE["a"] and parsE["a"]>1e-6:
            a=parsE["a"]; e=parsE["e"]; inc=parsE["inc"]; Om=parsE["Om"]; w=parsE["w"]; M=parsE["M"]
            import math as _m
            Mrad=_m.radians(M); incR=_m.radians(inc); OmR=_m.radians(Om); wR=_m.radians(w)
            E=Mrad if e<0.8 else _m.pi
            for _ in range(20):
                dE=(E - e*_m.sin(E) - Mrad)/(1 - e*_m.cos(E))
                E-=dE
                if abs(dE)<1e-12:
                    break
            cosE=_m.cos(E); sinE=_m.sin(E); sq=_m.sqrt(max(0,1-e*e))
            x_orb=a*(cosE - e); y_orb=a*sq*sinE
            n=_m.sqrt(4*_m.pi**2*0.998/(a*a*a+1e-18)) if a>1e-6 else 0
            vx_orb=-n*a*sinE/(1-e*cosE) if (1-e*cosE)!=0 else 0
            vy_orb=n*a*sq*_m.cos(E)/(1-e*cosE) if (1-e*cosE)!=0 else 0
            cosO=_m.cos(OmR); sinO=_m.sin(OmR); cosi=_m.cos(incR); sini=_m.sin(incR); cosw=_m.cos(wR); sinw=_m.sin(wR)
            x=(cosO*cosw - sinO*sinw*cosi)*x_orb + (-cosO*sinw - sinO*cosw*cosi)*y_orb
            y=(sinO*cosw + cosO*sinw*cosi)*x_orb + (-sinO*sinw + cosO*cosw*cosi)*y_orb
            z=(sinw*sini)*x_orb + (cosw*sini)*y_orb
            vx=(cosO*cosw - sinO*sinw*cosi)*vx_orb + (-cosO*sinw - sinO*cosw*cosi)*vy_orb
            vy=(sinO*cosw + cosO*sinw*cosi)*vx_orb + (-sinO*sinw + cosO*cosw*cosi)*vy_orb
            vz=(sinw*sini)*vx_orb + (cosw*sini)*vy_orb
            sim.add(m=mass_map.get(sid,1e-10), x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        else:
            parsV=parse_vectors(raw) if raw else None
            if parsV:
                sim.add(m=mass_map.get(sid,1e-10), x=parsV["x"], y=parsV["y"], z=parsV["z"], vx=parsV["vx"], vy=parsV["vy"], vz=parsV["vz"])
    # luas (VECTORS heliocêntrico direto, já inclui planeta)
    moon_ids=[]
    for lst in MOON_RANGES.values():
        moon_ids+=lst
    if limit:
        moon_ids=moon_ids[:max(0,(limit or 430)-len(PLANETS)-len(DWARFS)-1)]
    for sid in moon_ids:
        raw=bodies.get(str(sid),{}).get("raw","")
        pars=parse_vectors(raw)
        if not pars:
            # tenta ELEMENTS como fallback (planetocêntrico, mas VECTORS já é heliocêntrico)
            parsE=parse_elements(raw)
            if parsE and "a" in parsE:
                # planetocêntrico precisa somar planeta — simplifica: usa heliocêntrico VECTORS se falhar, ignora
                continue
            continue
        sim.add(m=1e-10, x=pars["x"], y=pars["y"], z=pars["z"], vx=pars["vx"], vy=pars["vy"], vz=pars["vz"])
    sim.move_to_com()
    return sim

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=10.0, help="anos para integrar")
    p.add_argument("--dt", type=float, default=0.01, help="dt para salvar (anos)")
    p.add_argument("--out", type=str, default="data/rebound_430.npz")
    p.add_argument("--limit", type=int, default=0, help="0=todos 430, ou N para teste")
    args=p.parse_args()
    lim=None if args.limit==0 else args.limit
    sim=build_rebound_sim_from_cache(limit=lim)
    print(f"N={sim.N} integrando {args.years} anos dt={args.dt} ...")
    t0=time.time()
    N=sim.N
    steps=int(args.years/args.dt)
    traj=np.zeros((steps+1, N, 3), dtype=np.float32)
    masses=np.array([p.m for p in sim.particles], dtype=np.float64)
    # salva labels aproximados: 0 Sol, 1-8 planetas, 9 luas, 10 anões
    labels=np.zeros(N, dtype=np.int32)
    # mapeia por ordem: 0 Sol, 1-8 planetas, resto luas/anões
    if N>9:
        labels[1:9]=np.arange(1,9)
        labels[9:]=9
        if N>430-5:
            labels[-5:]=10
    for i in range(N):
        traj[0,i]=[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z]
    for step in range(1, steps+1):
        sim.integrate(step*args.dt)
        for i in range(N):
            traj[step,i]=[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z]
        if step % max(1, steps//10)==0:
            print(f"  step {step}/{steps} t={sim.t:.2f} yr")
    elapsed=time.time()-t0
    print(f"Integrado {steps} steps em {elapsed:.1f}s ({elapsed/steps*1000:.1f} ms/step) N={N}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, traj=traj, masses=masses, labels=labels, dt=args.dt, years=args.years)
    print(f"Salvo {args.out} traj {traj.shape} {traj.nbytes/1e6:.1f} MB")
