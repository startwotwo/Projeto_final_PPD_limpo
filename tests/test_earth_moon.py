import json, pathlib, re, rebound, math
cache=json.loads(pathlib.Path('data/horizons_cache.json').read_text(encoding='utf-8'))
def parse_vectors(raw):
    m=re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
    txt=m.group(1).strip() if m else raw
    for line in txt.splitlines():
        line=line.strip()
        if not line or line.startswith('$$'):
            continue
        nums=re.findall(r'[-+]?\d+\.\d+E[+-]\d+', line)
        if len(nums)>=6:
            x=float(nums[0]); y=float(nums[1]); z=float(nums[2]); vx=float(nums[3]); vy=float(nums[4]); vz=float(nums[5])
            vx*=365.25; vy*=365.25; vz*=365.25
            return x*0.1, y*0.1, z*0.1, vx*0.1, vy*0.1, vz*0.1
    return None

for test_n in [1,5]:
    sim=rebound.Simulation()
    sim.units=("AU","yr","Msun")
    sim.G=4*math.pi**2
    sim.integrator="ias15"
    sim.add(m=1.0)
    for cid in [399,301]:
        raw=cache['bodies'].get(str(cid),{}).get("raw","")
        x,y,z,vx,vy,vz=parse_vectors(raw)
        m={399:3e-6,301:3.7e-8}[cid]
        sim.add(m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    sim.move_to_com()
    try:
        sim.integrate(0.5)
        print(f"Earth+Moon {test_n} ok", not any(p.x!=p.x for p in sim.particles))
    except Exception as e:
        print(f"fail {e}")
