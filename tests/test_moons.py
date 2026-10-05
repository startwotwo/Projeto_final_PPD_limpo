import rebound, json, re, urllib.request, urllib.parse, math
def fetch_one(cid, center="500@10"):
    import datetime as _dt
    d=_dt.datetime.fromisoformat('2026-01-01')
    stop=(d+_dt.timedelta(days=1)).date().isoformat()
    params={'format':'json','COMMAND':str(cid),'OBJ_DATA':'NO','MAKE_EPHEM':'YES','EPHEM_TYPE':'VECTORS','CENTER':center,'START_TIME':'2026-01-01','STOP_TIME':stop,'STEP_SIZE':'1d','OUT_UNITS':'AU-D','VEC_LABELS':'YES','CSV_FORMAT':'YES'}
    url='https://ssd.jpl.nasa.gov/api/horizons.api?'+urllib.parse.urlencode(params)
    data=json.loads(urllib.request.urlopen(url, timeout=20).read().decode())
    return data.get('result','')
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

def test_with_moons(moon_ids):
    sim=rebound.Simulation()
    sim.units=("AU","yr","Msun")
    sim.G=4*math.pi**2
    sim.integrator="ias15"
    sim.add(m=1.0)
    for cid in [199,299,399,499,599,699,799,899,2000001,134340,136199,136108,136472]:
        raw=fetch_one(cid)
        x,y,z,vx,vy,vz=parse_vectors(raw)
        mass={199:1.65e-7,299:2.44e-6,399:3e-6,499:3.21e-7,599:9.54e-4,699:2.85e-4,799:4.37e-5,899:5.15e-5,2000001:4.7e-10,134340:6.6e-9,136199:8.4e-9,136108:2.0e-9,136472:1.5e-9}.get(cid,1e-10)
        sim.add(m=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    for cid in moon_ids:
        # try heliocentric first
        raw=fetch_one(cid, "500@10")
        pars=parse_vectors(raw)
        if not pars:
            continue
        x,y,z,vx,vy,vz=pars
        sim.add(m=1e-10, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    sim.move_to_com()
    try:
        sim.integrate(0.5)
        has_nan=any(p.x!=p.x for p in sim.particles)
        return not has_nan
    except:
        return False

# Test batches
for n in [1,5,10,20,50,100]:
    moon_ids=list(range(501,501+n))
    ok=test_with_moons(moon_ids)
    print(f"n={n} {'ok' if ok else 'NaN'}")
    if not ok:
        break
