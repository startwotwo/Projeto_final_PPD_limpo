r"""
Merge de GM nos caches existentes (rápido, sem efeméride).
============================================================
Busca só OBJ_DATA=YES (sem MAKE_EPHEM) para cada ID já presente no cache
e grava bodies[id]["gm"] (km^3/s^2, null se n.a.). O preset usa GM quando
existe e cai para a tabela quando não (ex: luas novas, maioria dos asteroides).

Uso:
  .venv312\Scripts\python scripts/fetch_gm_merge.py --cache data/horizons_cache.json [--limit 5]
"""
import argparse
import json
import pathlib
import re
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_gm(command, timeout=20):
    params = {"format": "json", "COMMAND": str(command), "OBJ_DATA": "YES",
              "MAKE_EPHEM": "NO"}
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("result", "")


def parse_gm(raw):
    m = re.search(r'GM\s*\(km\^3/s\^2\)\s*=\s*([0-9.\-E+]+)', raw or "")
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'\bGM\s*=\s*([0-9.\-E+]+)', raw or "")
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def one(cid, delay, tries=5):
    for k in range(tries):
        try:
            if delay > 0:
                time.sleep(delay)
            gm = parse_gm(fetch_gm(cid))
            return cid, gm, None
        except Exception as e:
            err = str(e)[:100]
            time.sleep(2.0 * (2 ** k))  # backoff longo: 503 = JPL pedindo arrego
    return cid, None, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/horizons_cache.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.5, help="pausa por req (s)")
    p.add_argument("--force", action="store_true", help="refaz mesmo quem já tem gm")
    args = p.parse_args()

    cp = pathlib.Path(args.cache)
    cache = json.loads(cp.read_text(encoding="utf-8"))
    ids = list(cache.get("bodies", {}).keys())
    if not args.force:
        ids = [cid for cid in ids if cache["bodies"][cid].get("gm") is None]
    if args.limit:
        ids = ids[:args.limit]
    print(f"merge GM: {len(ids)} IDs em {cp} ({args.workers} workers, delay {args.delay}s)")

    have = 0
    with ThreadPoolExecutor(max_workers=min(max(1, args.workers), 3)) as ex:
        futs = {ex.submit(one, cid, args.delay): cid for cid in ids}
        for i, f in enumerate(as_completed(futs), 1):
            cid, gm, err = f.result()
            if gm is not None:
                cache["bodies"][cid]["gm"] = gm
                have += 1
            elif err:
                print(f"  {cid}: {err}")
            if i % 50 == 0 or i == len(ids):
                print(f"  [{i}/{len(ids)}] com GM: {have}", flush=True)
    import shutil
    shutil.copy(cp, str(cp) + ".pre-gm-bak")  # backup antes de sobrescrever
    cp.write_text(json.dumps(cache), encoding="utf-8")
    print(f"OK: {have}/{len(ids)} com GM -> {cp}")


if __name__ == "__main__":
    main()
