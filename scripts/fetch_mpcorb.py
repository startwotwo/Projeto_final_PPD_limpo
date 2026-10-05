r"""
MPCORB -> NPZ compacto (a fonte em massa p/ 1.3M de corpos reais).
====================================================================
Baixa o catálogo do Minor Planet Center (1 download, sem rate limit de API),
converte elementos heliocêntricos J2000 -> vetores de estado e salva NPZ
(desig, pos_au, vel_auyr, epoch, H) compatível com --asteroids/--asteroids-cache.

Colunas (MPC, doc Export Format for Minor-Planet Orbits):
  desig 0:7, H 8:13, época packed 20:25, M 26:35, argperi 37:46,
  node 48:57, incl 59:68, e 70:79, n(g/dia) 80:91, a(AU) 92:103,
  nome legível 166:194. Elementos heliocêntricos, eclíptica J2000.0
  (mesmo referencial dos VECTORS do Horizons).

Massas: o MPCORB NÃO traz massa/GM (só H). Aqui vai massa 0.0
(partícula-teste declarada, sem inventar dado). Diâmetro via H exigiria
albedo arbitrado — fora do escopo, H fica guardado p/ trabalho futuro.
Exceção: nº 1 (Ceres) é pulado (já é anão no sistema principal).

Uso:
  .venv312\Scripts\python scripts/fetch_mpcorb.py --out data/mpcorb.npz
  .venv312\Scripts\python scripts/fetch_mpcorb.py --out data/mpcorb.npz --max 5000  (teste)
  .venv312\Scripts\python scripts/live_vispy.py --preset solar_completo --asteroids 50000 --asteroids-cache data/mpcorb.npz --backend bh_gpu
"""
import argparse
import gzip
import pathlib
import shutil
import sys
import time
import urllib.request

import numpy as np

MPCORB_URL = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"

_CENT = {"I": 18, "J": 19, "K": 20}
_MONTH = {str(i): i for i in range(1, 10)}
_MONTH.update({"A": 10, "B": 11, "C": 12})
_DAY = {str(i): i for i in range(1, 10)}
_DAY.update({c: 10 + k for k, c in enumerate("ABCDEFGHIJKLMNOPQRSTUV")})


def unpack_epoch(s):
    """'K01AM' -> '2001-10-22'. Retorna None se inválido."""
    try:
        yyyy = _CENT[s[0]] * 100 + int(s[1:3])
        mm = _MONTH[s[3]]
        dd = _DAY[s[4]]
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return None


def download(url, dest):
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 10_000_000:
        print(f"usando {dest} existente ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"baixando {url} ...")
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 512)
    print(f"OK {dest.stat().st_size / 1e6:.1f} MB em {time.perf_counter() - t0:.0f}s")
    return dest


def parse_records(path, max_n=0):
    """Lê .DAT(.gz) -> dict de listas. Pula órbitas não-elípticas (conta)."""
    import gzip as _gz
    opener = _gz.open if str(path).endswith(".gz") else open
    a, e, inc, Om, w, M, n = [], [], [], [], [], [], []
    H, desig, ep_list, opp = [], [], [], []
    n_hyp = n_bad = n_ceres = 0
    total = 0
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if max_n and total >= max_n:
                break
            if len(line) < 103:
                continue
            total += 1
            try:
                num = line[0:7].strip()
                # anões do sistema principal (Ceres, Haumea, Makemake, Eris)
                # vêm no MPCORB com packed-letra ("D6108"), então checa o nome
                # legível em vez do número (Pluto não vem no MPCORB)
                nm0 = line[166:194].strip() if len(line) >= 194 else ""
                if nm0 in ("Ceres", "(1) Ceres", "Haumea", "(136108) Haumea",
                           "Makemake", "(136472) Makemake", "Eris", "(136199) Eris",
                           "Pluto", "(134340) Pluto") or (
                           num.isdigit() and int(num) in (1, 134340, 136108, 136199, 136472)):
                    n_ceres += 1
                    continue
                Hv = float(line[8:13])
                ep = unpack_epoch(line[20:25])
                if ep is None:
                    n_bad += 1
                    continue
                Mv = float(line[26:35])
                wv = float(line[37:46])
                Omv = float(line[48:57])
                iv = float(line[59:68])
                ev = float(line[70:79])
                nv = float(line[80:91])
                av = float(line[92:103])
                if not (np.isfinite(av) and np.isfinite(ev) and av > 0 and ev < 1.0):
                    n_hyp += 1
                    continue
                try:
                    op = int(line[123:126])
                except Exception:
                    op = 1
                nm = line[166:194].strip() if len(line) >= 194 else ""
                if not nm:
                    nm = f"({num})" if num else "(?)"
                a.append(av); e.append(ev); inc.append(iv); Om.append(Omv)
                w.append(wv); M.append(Mv); n.append(nv)
                H.append(Hv); desig.append(nm[:28]); ep_list.append(ep); opp.append(op)
            except Exception:
                n_bad += 1
                continue
    print(f"linhas: {total} ok={len(a)} hiperbólicas/e>=1={n_hyp} ruins={n_bad} ceres_dup={n_ceres}")
    return (np.array(a), np.array(e), np.array(inc), np.array(Om), np.array(w),
            np.array(M), np.array(n), np.array(H), desig, ep_list, opp)


def kepler_to_vectors(a, e, inc_d, Om_d, w_d, M_d, n_d):
    """Elementos (arrays) -> pos AU, vel AU/dia. Numba prange."""
    import numba

    @numba.njit(parallel=True)
    def _solve(a, e, inc_d, Om_d, w_d, M_d, n_d, pos, vel):
        nn = a.shape[0]
        for k in numba.prange(nn):
            Mk = np.radians(M_d[k] % 360.0)
            ek = e[k]
            E = Mk if ek < 0.8 else np.pi
            for _ in range(15):
                den = 1.0 - ek * np.cos(E)
                if abs(den) < 1e-14:
                    den = 1e-14
                dE = (E - ek * np.sin(E) - Mk) / den
                E -= dE
                if abs(dE) < 1e-12:
                    break
            ce = np.cos(E)
            se = np.sin(E)
            sq = np.sqrt(max(0.0, 1.0 - ek * ek))
            den2 = 1.0 - ek * ce
            if abs(den2) < 1e-12:
                den2 = 1e-12
            xo = a[k] * (ce - ek)
            yo = a[k] * sq * se
            nr = n_d[k] * np.pi / 180.0  # rad/dia
            vxo = -a[k] * nr * se / den2
            vyo = a[k] * nr * sq * ce / den2
            inc = np.radians(inc_d[k])
            Om = np.radians(Om_d[k])
            w = np.radians(w_d[k])
            cO = np.cos(Om); sO = np.sin(Om)
            ci = np.cos(inc); si = np.sin(inc)
            cw = np.cos(w); sw = np.sin(w)
            r11 = cO * cw - sO * sw * ci
            r12 = -cO * sw - sO * cw * ci
            r21 = sO * cw + cO * sw * ci
            r22 = -sO * sw + cO * cw * ci
            r31 = sw * si
            r32 = cw * si
            pos[k, 0] = r11 * xo + r12 * yo
            pos[k, 1] = r21 * xo + r22 * yo
            pos[k, 2] = r31 * xo + r32 * yo
            vel[k, 0] = r11 * vxo + r12 * vyo
            vel[k, 1] = r21 * vxo + r22 * vyo
            vel[k, 2] = r31 * vxo + r32 * vyo

    pos = np.zeros((a.shape[0], 3), dtype=np.float64)
    vel = np.zeros((a.shape[0], 3), dtype=np.float64)
    _solve(a, e, inc_d, Om_d, w_d, M_d, n_d, pos, vel)
    return pos, vel


def main():
    from collections import Counter
    p = argparse.ArgumentParser(description="MPCORB -> NPZ")
    p.add_argument("--out", default="data/mpcorb.npz")
    p.add_argument("--dat", default="data/MPCORB.DAT.gz")
    p.add_argument("--url", default=MPCORB_URL)
    p.add_argument("--max", type=int, default=0, help="limita linhas (teste)")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--min-oppositions", type=int, default=1,
                   help="descarta órbitas piores que N oposições (1 = tudo)")
    p.add_argument("--propagate-to", default=None,
                   help="avança M por n*dias até YYYY-MM-DD (Kepler 2-corpos exato; "
                        "sem perturbações planetárias: arco-minutos de erro em meses)")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dat = pathlib.Path(args.dat)
    if not args.no_download:
        download(args.url, dat)

    t0 = time.perf_counter()
    a, e, inc, Om, w, M, n, H, desig, ep_list, opp = parse_records(dat, args.max)
    print(f"parse: {len(a)} corpos em {time.perf_counter() - t0:.0f}s")
    if not len(a):
        print("nada parseado, abortando")
        sys.exit(1)
    # época DOMINANTE (o arquivo mistura órbitas velhas não-atualizadas;
    # usar a mais antiga = incoerente). Filtra todo o resto.
    cnt = Counter(ep_list)
    epoch = cnt.most_common(1)[0][0]
    print(f"top épocas: {cnt.most_common(5)}")
    keep = np.array([ep == epoch for ep in ep_list])
    dropped_epoch = int((~keep).sum())
    # qualidade: oposições
    opp = np.array(opp)
    keep = keep & (opp >= args.min_oppositions)
    dropped_opp = int(((np.array(ep_list) == epoch) & (opp < args.min_oppositions)).sum())
    a, e, inc, Om, w, M, n, H = a[keep], e[keep], inc[keep], Om[keep], w[keep], M[keep], n[keep], H[keep]
    desig = [d for d, k in zip(desig, keep) if k]
    print(f"época {epoch}: {len(a)} corpos (fora de época: {dropped_epoch}, "
          f"<{args.min_oppositions} oposições: {dropped_opp})")
    if args.propagate_to:
        import datetime as _dt
        dd = (_dt.datetime.fromisoformat(args.propagate_to) - _dt.datetime.fromisoformat(epoch)).total_seconds() / 86400.0
        M = (M + n * dd) % 360.0  # mesma órbita, só anda na anomalia média (exato em 2-corpos)
        print(f"propagado {epoch} -> {args.propagate_to} ({dd:.1f} dias, 2-corpos, sem perturbações)")
        epoch = args.propagate_to

    t0 = time.perf_counter()
    pos, vel = kepler_to_vectors(a, e, inc, Om, w, M, n)
    print(f"kepler: {len(a)} vetores em {time.perf_counter() - t0:.1f}s")
    np.savez_compressed(
        out,
        desig=np.array(desig, dtype="<U28"),
        pos_au=pos,
        vel_auyr=vel * 365.25,
        epoch=np.array(epoch),
        H=np.array(H),
    )
    print(f"OK: {out} ({len(a)} corpos, época {epoch})")


if __name__ == "__main__":
    main()
