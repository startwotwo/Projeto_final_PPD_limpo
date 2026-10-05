r"""
Fetch paralelo de asteroides numerados (JPL Horizons) -> NPZ compacto.
==============================================================================
Baixa VECTORS heliocêntricos (500@10) dos asteroides numerados  --start..--start+--count
e salva arrays compactos (sem raw): ids, pos_au, vel_auyr + época.

Por que não os 1.3M: a 5-10 req/s, 1.3M = 1.5-3 MESES contínuos e o JPL bloqueia
abuso. Amostras realistas: 10k (~30-60 min, 4-8 workers), 100k (~5-10 h).
Respeite o serviço: workers<=8, delay entre reqs, retry com backoff.

Uso:
  .venv312\Scripts\python scripts/fetch_asteroids.py --start 1 --count 10000 --out data/asteroids.npz --epoch 2026-09-28 --workers 6
  # retomar de onde parou (lê o NPZ existente e pula IDs já baixados):
  .venv312\Scripts\python scripts/fetch_asteroids.py --start 1 --count 100000 --out data/asteroids.npz --resume

Depois:
  python scripts/live_vispy.py --preset solar_completo --asteroids 10000
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def fetch_one(num, epoch, timeout=30):
    """VECTORS heliocêntrico do asteroide numerado `num` (COMMAND='N;')."""
    import datetime as _dt
    try:
        d = _dt.datetime.fromisoformat(epoch)
        stop = (d + _dt.timedelta(days=1)).date().isoformat()
    except Exception:
        stop = epoch
    params = {
        "format": "json",
        "COMMAND": f"{num};",  # ';' = asteroide numerado (sem ambiguidade c/ corpo maior)
        "OBJ_DATA": "YES",  # traz GM junto (NaN p/ maioria); sem request extra
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "500@10",
        "START_TIME": epoch,
        "STOP_TIME": stop,
        "STEP_SIZE": "1d",
        "OUT_UNITS": "AU-D",
        "VEC_LABELS": "YES",
        "CSV_FORMAT": "YES",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("result", "")


def parse_vectors_au(raw):
    """Extrai (pos AU, vel AU/ano, GM km^3/s^2 ou NaN) da resposta."""
    m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
    if not m:
        return None
    gm = re.search(r'GM\s*\(km\^3/s\^2\)\s*=\s*([0-9.\-E+]+)', raw)
    if gm is None:
        gm = re.search(r'\bGM\s*=\s*([0-9.\-E+]+)', raw)
    gm = float(gm.group(1)) if gm else float("nan")
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line or line.startswith("$$"):
            continue
        nums = re.findall(r'[-+]?\d+\.\d+E[+-]\d+', line)
        if len(nums) >= 6:
            try:
                x = np.array([float(nums[0]), float(nums[1]), float(nums[2])])
                v = np.array([float(nums[3]), float(nums[4]), float(nums[5])]) * 365.25
                return x, v, gm
            except Exception:
                continue
    return None


def fetch_with_retry(num, epoch, delay, tries=3):
    for k in range(tries):
        try:
            if delay > 0:
                time.sleep(delay)
            raw = fetch_one(num, epoch)
            pv = parse_vectors_au(raw)
            if pv is not None:
                return num, pv[0], pv[1], pv[2], None
            # sem VECTORS (nº vago/erro): não é retry, é skip
            if "No ephemeris" in raw or "No matches" in raw or len(raw) < 500:
                return num, None, None, None, "skip"
            return num, None, None, None, "noparse"
        except Exception as e:
            err = str(e)[:120]
            time.sleep(0.5 * (2 ** k))  # backoff (429/lockout)
    return num, None, None, None, f"fail: {err}"


def main():
    p = argparse.ArgumentParser(description="Fetch paralelo de asteroides -> NPZ")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--out", default="data/asteroids.npz")
    p.add_argument("--epoch", default="2026-09-28")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--delay", type=float, default=0.05, help="pausa por req/worker (s)")
    p.add_argument("--resume", action="store_true", help="pula IDs já presentes no NPZ")
    p.add_argument("--save-every", type=int, default=500)
    args = p.parse_args()

    if args.workers > 8:
        print("[aviso] workers>8 pode levar a ban do JPL; usando 8")
        args.workers = 8

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids_done, pos_done, vel_done, gm_done = [], [], [], []
    if args.resume and out.exists():
        z = np.load(out, allow_pickle=True)
        ids_done = [int(i) for i in z["ids"]]
        pos_done = [np.array(r) for r in z["pos_au"]]
        vel_done = [np.array(r) for r in z["vel_auyr"]]
        try:
            gm_done = [float(g) for g in z["gm"]]
        except Exception:
            gm_done = [float("nan")] * len(ids_done)  # NPZ antigo sem GM
        print(f"resume: {len(ids_done)} já baixados, pulando")

    have = set(ids_done)
    todo = [n for n in range(args.start, args.start + args.count) if n not in have]
    print(f"fetch: {len(todo)} asteroides ({args.start}..{args.start + args.count - 1}), "
          f"{args.workers} workers, época {args.epoch}")

    ok = skip = fail = 0
    t0 = time.perf_counter()
    buf = []

    def save():
        if not buf and not ids_done:
            return
        all_ids = ids_done + [b[0] for b in buf]
        all_pos = pos_done + [b[1] for b in buf]
        all_vel = vel_done + [b[2] for b in buf]
        all_gm = gm_done + [b[3] for b in buf]
        order = np.argsort(all_ids)
        np.savez_compressed(
            out,
            ids=np.array([all_ids[i] for i in order]),
            pos_au=np.vstack([all_pos[i] for i in order]),
            vel_auyr=np.vstack([all_vel[i] for i in order]),
            gm=np.array([all_gm[i] for i in order], dtype=np.float64),
            epoch=np.array(args.epoch),
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_with_retry, n, args.epoch, args.delay): n for n in todo}
            done = 0
            for f in as_completed(futs):
                num, x, v, gm, err = f.result()
                done += 1
                if x is None:
                    if err == "skip":
                        skip += 1
                    else:
                        fail += 1
                        if fail < 10:
                            print(f"  {num}: {err}")
                else:
                    ok += 1
                    buf.append((num, x, v, gm))
                if done % args.save_every == 0 or done == len(todo):
                    save()
                if done % 100 == 0 or done == len(todo):
                    el = time.perf_counter() - t0
                    rate = done / max(el, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-6) / 60
                    print(f"  [{done}/{len(todo)}] ok={ok} skip={skip} fail={fail} "
                          f"{rate:.1f} req/s ETA {eta:.0f}min", flush=True)
    except KeyboardInterrupt:
        print("\ninterrompido! salvando parcial...")
    save()
    print(f"OK: {out} ({ok} novos, total {len(ids_done) + len(buf)} corpos)")


if __name__ == "__main__":
    main()
