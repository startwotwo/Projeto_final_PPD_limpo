def build_simulation():
    import numpy as np
    import json
    import math
    
    # Load cache first
    cache_data = json.loads(open("data/horizons_cache.json", "r", encoding="utf-8").read())
    bodies_cache = cache["bodies"]
    
    # Known masses (Msun)
    mass_map = {
        199: 1.65e-7, 299: 2.44e-6, 399: 3.00e-6, 499: 3.21e-7,
        599: 9.54e-4, 699: 2.85e-4, 799: 4.37e-5, 899: 5.15e-5,
        1: 1.0,  # Sun
        134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9,
    }
    
    # Planet IDs
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    
    # Load cache
    cache_data = json.loads(open("data/horizons_cache.json", "r", encoding="utf-8").read())
    bodies_cache = cache["bodies"]
    
    # Parse function for raw data
    def parse_elements_raw(sid):
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        txt = m.group(1) if m else raw
        def _get(key):
            m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
            return float(m.group(1)) if m else None
        ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
        Om = _get("OM"); w = _get("W"); ma = _get("MA")
        if a_km is None:
            return None
        a_au = a_km / 149597870.7
        a = a_au / 10.0
        return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
    
    def true_anomaly_from_M(e, M_deg):
        M = np.radians(M_deg)
        E = M if e < 0.8 else np.pi
        for _ in range(30):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return f
    
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.G = 4*np.pi**2
    sim.integrator = "ias15"
    
    # Known masses (Msun)
    mass_map = {
        199: 1.65e-7, 299: 2.44e-6, 399: 3.00e-6, 499: 3.21e-7,
        599: 9.54e-4, 699: 2.85e-4, 799: 4.37e-5, 899: 5.15e-5,
        1: 1.0,  # Sun
        134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9,
    }
    
    # Planet IDs
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    
    # Parse function for raw data
    def parse_elements_raw(sid):
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        txt = m.group(1) if m else raw
        def _get(key):
            m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
            return float(m.group(1)) if m else None
        ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
        Om = _get("OM"); w = _get("W"); ma = _get("MA")
        if a_km is None:
            return None
        a_au = a_km / 149597870.7
        a = a_au / 10.0
        return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
    
    def true_anomaly_from_M(e, M_deg):
        M = np.radians(M_deg)
        E = M if e < 0.8 else np.pi
        for _ in range(30):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return f
    
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.G = 4*np.pi**2
    sim.integrator = "ias15"
    
    # Add Sun
    sim.add(m=1.0)
    
    # Known masses (Msun)
    mass_map = {
        199: 1.65e-7, 299: 2.44e-6, 399: 3.00e-6, 499: 3.21e-7,
        599: 9.54e-4, 699: 2.85e-4, 799: 4.37e-5, 899: 5.15e-5,
        1: 1.0,  # Sun
        134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9,
    }
    
    # Planet IDs
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    
    # Load cache
    cache_data = json.loads(open("data/horizons_cache.json", "r", encoding="utf-8").read())
    bodies_cache = cache["bodies"]
    
    # Parse function for raw data
    def parse_elements_raw(sid):
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        txt = m.group(1) if m else raw
        def _get(key):
            m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
            return float(m.group(1)) if m else None
        ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
        Om = _get("OM"); w = _get("W"); ma = _get("MA")
        if a_km is None:
            return None
        a_au = a_km / 149597870.7
        a = a_au / 10.0
        return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
    
    def true_anomaly_from_M(e, M_deg):
        M = np.radians(M_deg)
        E = M if e < 0.8 else np.pi
        for _ in range(30):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return f
    
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.G = 4*np.pi**2
    sim.integrator = "ias15"
    
    # Add Sun
    sim.add(m=1.0)
    
    # Known masses (Msun)
    mass_map = {
        199: 1.65e-7, 299: 2.44e-6, 399: 3.00e-6, 499: 3.21e-7,
        599: 9.54e-4, 699: 2.85e-4, 799: 4.37e-5, 899: 5.15e-5,
        1: 1.0,  # Sun
        134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9,
    }
    
    # Planet IDs
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    
    # Parse function for raw data
    def parse_elements_raw(sid):
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        txt = m.group(1) if m else raw
        def _get(key):
            m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
            return float(m.group(1)) if m else None
        ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
        Om = _get("OM"); w = _get("W"); ma = _get("MA")
        if a_km is None:
            return None
        a_au = a_km / 149597870.7
        a = a_au / 10.0
        return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
    
    def true_anomaly_from_M(e, M_deg):
        M = np.radians(M_deg)
        E = M if e < 0.8 else np.pi
        for _ in range(30):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return f
    
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.G = 4*np.pi**2
    sim.integrator = "ias15"
    
    # Add Sun
    sim.add(m=1.0)
    
    # Known masses (Msun)
    mass_map = {
        199: 1.65e-7, 299: 2.44e-6, 399: 3.00e-6, 499: 3.21e-7,
        599: 9.54e-4, 699: 2.85e-4, 799: 4.37e-5, 899: 5.15e-5,
        1: 1.0,  # Sun
        134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9,
    }
    
    # Planet IDs
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    
    # Load cache
    cache_data = json.loads(open("data/horizons_cache.json", "r", encoding="utf-8").read())
    bodies_cache = cache["bodies"]
    
    # Parse function for raw data
    def parse_elements_raw(sid):
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        txt = m.group(1) if m else raw
        def _get(key):
            m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
            return float(m.group(1)) if m else None
        ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
        Om = _get("OM"); w = _get("W"); ma = _get("MA")
        if a_km is None:
            return None
        a_au = a_km / 149597870.7
        a = a_au / 10.0
        return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
    
    def true_anomaly_from_M(e, M_deg):
        M = np.radians(M_deg)
        E = M if e < 0.8 else np.pi
        for _ in range(30):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return f
    
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.G = 4*np.pi**2
    sim.integrator = "ias15"
    
    # Add Sun
    sim.add(m=1.0)
    
    # Known masses (Msun)
    mass_map = {
        199: 1.65e-7, 299: 2.44e-6, 399: 3.00e-6, 499: 3.21e-7,
        599: 9.54e-4, 699: 2.85e-4, 799: 4.37e-5, 899: 5.15e-5,
        1: 1.0,  # Sun
        134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9,
    }
    
    # Planet IDs
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    
    # Load cache
    cache_data = json.loads(open("data/horizons_cache.json", "r", encoding="utf-8").read())
    bodies_cache = cache["bodies"]
    
    # Parse function for raw data
    def parse_elements_raw(sid):
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        txt = m.group(1) if m else raw
        def _get(key):
            m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
            return float(m.group(1)) if m else None
        ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
        Om = _get("OM"); w = _get("W"); ma = _get("MA")
        if a_km is None:
            return None
        a_au = a_km / 149597870.7
        a = a_au / 10.0
        return {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
    
    def true_anomaly_from_M(e, M_deg):
        M = np.radians(M_deg)
        E = M if e < 0.8 else np.pi
        for _ in range(30):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return f
    
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.G = 4*np.pi**2
    sim.integrator = "ias15"
    
    # Add Sun
    sim.add(m=1.0)
    
    # Add planets
    for sid in [199, 299, 399, 499, 599, 699, 799, 899]:
        raw = bodies_cache.get(str(sid), {}).get("raw", "")
        m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
        if m:
            txt = m.group(1)
            def _get(key):
                m = re.search(rf'\b{key}\s*=\s*([0-9\.\-E\+]+)', raw)
                return float(m.group(1)) if m else None
            ec = _get("EC"); a_km = _get("A"); inc = _get("IN")
            Om = _get("OM"); w = _get("W"); ma = _get("MA")
            if a_km is not None:
                a_au = a_km / 149597870.7
                a = a_au / 10.0
                pars = {"a": a, "e": ec, "inc": inc, "Om": Om, "w": w, "M": ma}
                e = pars["e"]
                M_deg = pars["M"]
                f = true_anomaly_from_M(e, M_deg)
                try:
                    sim.add(
                        m=mass_map.get(sid, 1e-10),
                        a=pars["a"], e=pars["e"], inc=np.radians(pars["inc"]),
                        Omega=np.radians(pars["Om"]), omega=np.radians(pars["w"]),
                        f=f
                    )
                except Exception as e:
                    print(f"Failed to add planet {sid}: {e}")
                    continue

    # Move to COM
    sim.move_to_com()
    
    return sim
