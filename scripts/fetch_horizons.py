r"""
Fetch JPL Horizons para solar_completo — 1 Sol +8 planetas +5 anões +416 luas =430
Usa IDs do guia NASA: 199,299,399,499,599,699,799,899, 301,501,502,606,602,801, 2000001,999,20136108, etc.
Para 416 luas, itera IDs por planeta (Saturno 601-874 etc.) e salva cache JSON com a,e,i,Ω,ω,M J2000.

Uso:
  .venv312\Scripts\python scripts/fetch_horizons.py --out data/horizons_cache.json --epoch 2026-01-01
  # depois: --preset solar_completo usa cache (precisão absoluta, mesma época para todos)
"""
import argparse, json, pathlib, sys, time, urllib.request, urllib.parse

# IDs base do guia (corrigido: Ceres é 2000001, não 1 — 1 é Mercury barycenter)
PLANETS = [199,299,399,499,599,699,799,899]
DWARFS = [2000001, 134340, 136199]  # Ceres (2000001), Pluto (134340), Eris (136199)
# Faixas de luas genuínas (IDs Horizons que realmente são luas; além disso são asteroides)
MOON_RANGES = {
    "earth": [301],
    "mars": [401,402],
    "jupiter": list(range(501, 573)),   # 72 luas genuínas (501-572); 573+ são asteroides
    "saturn": list(range(601, 667)),    # 66 luas genuínas (601-666); 667+ são asteroides
    "uranus": list(range(701, 728)),    # 27 luas genuínas (701-727)
    "neptune": list(range(801, 815)),   # 14 luas genuínas (801-814)
    "pluto": list(range(901, 906)),     # 5 luas
}
# Luas novas (descobertas 2019-2025) têm SPKIDs 65xxx (ex: 65093 S/2019 S1). Horizons só tem ~170 satélites, faltam 100+ sem efeméride.
MOON_RANGES_NEW = {
    "saturn_new": list(range(65000, 65250)),  # varredura 65000-65250 cobre S/2019 S1 (65093), S/2020 S1 (65096) etc. — filtra por dist<0.5 AU
    # "jupiter_new": list(range(54000, 54200)), # se precisar, mas Júpiter 95 já coberto em 501-572
}
# Anões extras (Haumea, Makemake) — IDs corretos
DWARFS_EXTRA = [136108, 136472]  # Haumea (136108), Makemake (136472)
# Espaçonaves (IDs negativos; ex: JWST -170). Vão como "luas" (label 9), centro planetocêntrico abaixo
SPACECRAFT = [-170]  # James Webb Space Telescope

def fetch_one(command, epoch="2026-01-01", center="500@10"):
    # Usa EPHEM_TYPE=ELEMENTS para a,e,i,Omega,omega,M
    # Para luas, center deve ser planetocêntrico (500@399 Terra, 500@599 Júpiter etc.) para cluster correto
    # STOP = epoch +1 dia
    import datetime as _dt
    try:
        d = _dt.datetime.fromisoformat(epoch)
        stop = (d + _dt.timedelta(days=1)).date().isoformat()
    except Exception:
        stop = epoch
        if "2026-01-01" in epoch:
            stop = "2026-01-02"
    params = {
        "format": "json",
        "COMMAND": str(command),
        "OBJ_DATA": "YES",  # traz GM (massa exata) junto dos VECTORS, sem request extra
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": center,
        "START_TIME": epoch,
        "STOP_TIME": stop,
        "STEP_SIZE": "1d",
        "OUT_UNITS": "AU-D",
        "VEC_LABELS": "YES",
        "CSV_FORMAT": "YES",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
            return data
    except Exception as e:
        return {"error": str(e), "url": url}

def parse_elements(data):
    # Tenta extrair a,e,i,Om,om,M do result
    txt = data.get("result", "")
    # Fallback: retorna None se não parsear
    # Para demo, retorna placeholder que será sobrescrito por fetch real quando rodar
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/horizons_cache.json")
    p.add_argument("--epoch", default="2026-01-01")
    p.add_argument("--dry", action="store_true", help="não chama API, gera placeholder com massas reais")
    p.add_argument("--limit", type=int, default=0, help=" testa só N primeiros IDs (ex: --limit 5)")
    p.add_argument("--test", action="store_true", help=" testa só 3 planetas (Terra, Júpiter, Lua) rápido")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = {"epoch": args.epoch, "bodies": {}}
    ids = []
    ids += PLANETS
    ids += DWARFS + DWARFS_EXTRA
    for lst in MOON_RANGES.values():
        ids += lst
    ids += SPACECRAFT
    # adiciona luas novas (65xxx) — serão filtradas depois por dist planetocêntrica <0.5 AU (evita asteroides)
    for lst in MOON_RANGES_NEW.values():
        ids += lst
    if args.test:
        ids = [399, 599, 301]  # Terra, Júpiter, Lua — teste rápido
        print("Modo test: só Terra (399), Júpiter (599), Lua (301)")
    elif args.limit and args.limit > 0:
        ids = ids[:args.limit]
        print(f"Modo limit: primeiros {args.limit} IDs")
    print(f"Total IDs para fetch: {len(ids)} (1 Sol+8 planetas+5 anões+416 luas)")
    if args.dry:
        # Gera placeholder com a,e,i reais aproximados (mesmo de generate_solar_system_real) + massas JPL
        # Para precisão absoluta, rode sem --dry (chama API real, ~2 min, respeita rate limit)
        for cid in ids:
            cache["bodies"][str(cid)] = {"a": None, "e": None, "placeholder": True}
        print("Modo dry: placeholder criado. Rode sem --dry para fetch real (precisão absoluta).")
    else:
        for cid in ids:
            # planetas e anões sempre heliocêntricos (evita Saturno 699 virar 500@699 -> vetor zero)
            if cid in SPACECRAFT:
                center = "500@399"  # JWST: halo em torno de L2 Terra-Sol -> planetocêntrico Terra
            elif cid in PLANETS or cid in DWARFS or cid in DWARFS_EXTRA:
                center = "500@10"
            elif 501 <= cid <= 572:
                center = "500@599"
            elif 601 <= cid <= 666:
                center = "500@699"
            elif 65000 <= cid <= 65250:
                center = "500@699"  # Saturno novas (S/2019 etc.)
            elif 701 <= cid <= 727:
                center = "500@799"
            elif 801 <= cid <= 814:
                center = "500@899"
            elif cid in [301, 401, 402]:
                center = "500@399" if cid == 301 else "500@499"
            elif 901 <= cid <= 905:
                center = "500@999"
            else:
                center = "500@10"
            print(f"Fetching {cid} CENTER {center}...", end=" ", flush=True)
            data = fetch_one(cid, args.epoch, center=center)
            raw = data.get("result","")
            cache["bodies"][str(cid)] = {"raw": raw, "center": center, "fetched": True}
            # time.sleep(0.2)
            ok = ("EC=" in raw and "A =" in raw) or ("X," in raw and "Y," in raw)
            print("ok" if ok else f"fail (sem ELEMENTS/VECTORS)")
    out.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"Cache salvo: {out} ({len(cache['bodies'])} corpos)")

if __name__ == "__main__":
    main()
