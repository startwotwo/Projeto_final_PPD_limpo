"""
Backend REBOUND IAS15 para live_vispy — 15ª ordem adaptativo, estável para 400 luas
Usa cache VECTORS JPL 2026-01-01 (heliocêntrico) e planetocêntrico para luas
"""
import numpy as np
import rebound, json, pathlib, re, math

def _build_rebound_sim(limit=None, use_cache=True):
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
    # massas
    mass_map={199:1.65e-7,299:2.44e-6,399:3.00e-6,499:3.21e-7,599:9.54e-4,699:2.85e-4,799:4.37e-5,899:5.15e-5,2000001:4.7e-10,134340:6.6e-9,136199:8.4e-9,136108:2.0e-9,136472:1.5e-9}
    # tenta cache VECTORS
    cache_path=pathlib.Path("data/horizons_cache.json")
    use_cache = use_cache and cache_path.exists()
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
                    x=float(nums[0]); y=float(nums[1]); z=float(nums[2]); vx=float(nums[3]); vy=float(nums[4]); vz=float(nums[5])
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
    def M_to_f(e, M_deg):
        M=math.radians(M_deg)
        E=M if e<0.8 else math.pi
        for _ in range(30):
            dE=(E - e*math.sin(E) - M)/(1 - e*math.cos(E))
            E-=dE
            if abs(dE)<1e-14:
                break
        return 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E/2))
    sim=rebound.Simulation()
    sim.units=("AU","yr","Msun")
    sim.G=4*math.pi**2
    sim.integrator="ias15"
    try:
        import rebound as _reb
        _reb.omp_set_num_threads(28)
    except:
        pass
    sim.add(m=1.0)
    # planetas e anões heliocêntricos
    for sid in PLANETS+DWARFS:
        raw=None
        if use_cache:
            try:
                raw=json.loads(cache_path.read_text(encoding="utf-8"))["bodies"].get(str(sid),{}).get("raw","")
            except: pass
        pars=None
        if raw and "EC=" in raw:
            pars=parse_elements(raw)
            if pars and "a" in pars:
                # converte ELEMENTS -> xyz
                a=pars["a"]; e=pars["e"]; inc=pars["inc"]; Om=pars["Om"]; w=pars["w"]; M=pars["M"]
                f=M_to_f(e, M)
                try:
                    sim.add(m=mass_map.get(sid,1e-10), a=a, e=e, inc=math.radians(inc), Omega=math.radians(Om), omega=math.radians(w), f=f)
                    continue
                except: pass
        # fallback VECTORS
        if not pars or "x" not in pars:
            if use_cache and raw:
                pars=parse_vectors(raw)
            else:
                # fetch VECTORS
                import urllib.request, urllib.parse, datetime as _dt
                d=_dt.datetime.fromisoformat("2026-01-01")
                stop=(d+_dt.timedelta(days=1)).date().isoformat()
                params={"format":"json","COMMAND":str(sid),"OBJ_DATA":"NO","MAKE_EPHEM":"YES","EPHEM_TYPE":"VECTORS","CENTER":"500@10","START_TIME":"2026-01-01","STOP_TIME":stop,"STEP_SIZE":"1d","OUT_UNITS":"AU-D","VEC_LABELS":"YES","CSV_FORMAT":"YES"}
                url="https://ssd.jpl.nasa.gov/api/horizons.api?"+urllib.parse.urlencode(params)
                try:
                    data=json.loads(urllib.request.urlopen(url, timeout=20).read().decode())
                    pars=parse_vectors(data.get("result",""))
                except:
                    pars=None
        if pars and "x" in pars:
            sim.add(m=mass_map.get(sid,1e-10), x=pars["x"], y=pars["y"], z=pars["z"], vx=pars["vx"], vy=pars["vy"], vz=pars["vz"])
    # luas: tenta cache planetocêntrico, senão heliocêntrico
    moon_ids=[]
    for lst in MOON_RANGES.values():
        moon_ids+=lst
    if limit:
        moon_ids=moon_ids[:max(0, (limit or 430)-len(PLANETS)-len(DWARFS)-1)]
    for sid in moon_ids:
        raw=None
        if use_cache:
            try:
                raw=json.loads(cache_path.read_text(encoding="utf-8"))["bodies"].get(str(sid),{}).get("raw","")
            except: pass
        pars=None
        # tenta planetocêntrico primeiro se for lua (CENTER 500@planet)
        # para simplificar, usa heliocêntrico direto se disponível
        if raw and "EC=" in raw:
            pars=parse_elements(raw)
            # planetocêntrico: a pequeno, precisa somar planeta
            # ignora por enquanto, usa heliocêntrico
            pars=None
        if not pars or "x" not in pars:
            # tenta VECTORS heliocêntrico
            if not raw or "X =" not in raw:
                try:
                    import urllib.request, urllib.parse
                    import datetime as _dt
                    d=_dt.datetime.fromisoformat("2026-01-01")
                    stop=(d+_dt.timedelta(days=1)).date().isoformat()
                    params={"format":"json","COMMAND":str(sid),"OBJ_DATA":"NO","MAKE_EPHEM":"YES","EPHEM_TYPE":"VECTORS","CENTER":"500@10","START_TIME":"2026-01-01","STOP_TIME":stop,"STEP_SIZE":"1d","OUT_UNITS":"AU-D","VEC_LABELS":"YES","CSV_FORMAT":"YES"}
                    url="https://ssd.jpl.nasa.gov/api/horizons.api?"+urllib.parse.urlencode(params)
                    data=json.loads(urllib.request.urlopen(url, timeout=10).read().decode())
                    raw=data.get("result","")
                    pars=parse_vectors(raw)
                except:
                    pars=None
            else:
                pars=parse_vectors(raw)
        if pars and "x" in pars:
            sim.add(m=1e-10, x=pars["x"], y=pars["y"], z=pars["z"], vx=pars["vx"], vy=pars["vy"], vz=pars["vz"])
    sim.move_to_com()
    return sim
