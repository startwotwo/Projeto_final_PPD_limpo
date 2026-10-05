#!/usr/bin/env python
"""
Benchmark N^2 em GPU vs Barnes-Hut em GPU — alta escala, direto ao crossover
=============================================================================
Compara forca bruta contra hierarquico ate cada um bater no seu muro:
  - gpu_persist (CuPy N^2, buffers persistentes): ate 5M (~81 s/step);
    10M = 325 s/step (probe de 1 step em probe_limits.json) -> impraticavel.
  - bh_gpu      (hibrido: build CPU + traversal GPU): ate 10M (~23.5 s/step);
    20M estoura VRAM de 8 GB + RAM do sistema -> fora.
  - bh_gpu_full (build 100% no device): ate 10M (~0.68 s/step);
    20M so com swap PCIe (ponto avulso pre-existente, nao re-executado).
Preset galaxy (seed 42). Metrica: ms/step (leapfrog manual, warmup fora do
cronometro, sem energia no laco). Plots: linear + log-log com inclinacoes de
referencia O(N^2)/O(N log N) e linha de crossover anotada.

Uso:
  .\\.venv312\\Scripts\\python scripts/benchmark_gpu_vs_bh.py [--steps 100]
  .\\.venv312\\Scripts\\python scripts/benchmark_gpu_vs_bh.py --plot-only benchmark_results/gpu_vs_bh_xxx.json [--merge benchmark_results/probe_limits.json]
"""
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import src.presets as presets
import src.presets_runner_fast as runner
from run_benchmarks import BACKEND_COLORS

# Grade comum (comparacao ponto a ponto) + teto por backend.
N_COMMON = [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000,
            350000, 500000, 1000000, 2000000, 5000000]
N_PER_BACKEND = {
    "gpu_persist": N_COMMON,                       # 10M entra via --merge (probe)
    "bh_gpu":      N_COMMON + [10000000],          # 10M entra via --merge (probe)
    "bh_gpu_full": N_COMMON + [10000000],
}
BACKENDS = list(N_PER_BACKEND.keys())

LABELS = {
    "gpu_persist": "gpu_persist — N² direto (CuPy)",
    "bh_gpu":      "bh_gpu — Barnes-Hut híbrido (build CPU + trav. GPU)",
    "bh_gpu_full": "bh_gpu_full — Barnes-Hut 100% device",
}

DT = 0.005
EPS = 0.05

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = OUTPUT_DIR / f"gpu_vs_bh_{TIMESTAMP}.json"


def get_steps(N, base_steps):
    # N muito grande: menos steps (ms/step continua comparavel)
    if N >= 1000000:
        return 3
    if N >= 100000:
        return min(base_steps, 10)
    return base_steps


def run_one(backend, N, steps):
    pos, vel, masses, _ = presets.generate_spiral_galaxy(
        N_disk=int(N * 0.7), N_bulge=N - int(N * 0.7), seed=42
    )
    if backend == "bh_gpu_full":
        from src.nbody_bh_gpu_build import compute_forces_bh_gpu_device as _cfd
        def cf(pos, masses, G=1.0, eps=EPS):
            return _cfd(pos, masses, G, eps)
    else:
        cf, _, _ = runner._get_backends(backend)
    # warmup (compilacao + cache persistente) fora do cronometro
    cf(pos, masses, 1.0, EPS)
    pos_c = pos.copy()
    vel_c = vel.copy()
    accel = cf(pos_c, masses, 1.0, EPS)
    t0 = time.perf_counter()
    for _ in range(steps):
        vel_half = vel_c + accel * (DT * 0.5)
        pos_c = pos_c + vel_half * DT
        accel = cf(pos_c, masses, 1.0, EPS)
        vel_c = vel_half + accel * (DT * 0.5)
    t_total = time.perf_counter() - t0
    return {
        "backend": backend,
        "N": N,
        "steps": steps,
        "total_time_sec": t_total,
        "ms_per_step": (t_total / steps) * 1000,
        "success": True,
        "error": None,
    }


def run_all(base_steps):
    results = []
    total = sum(len(v) for v in N_PER_BACKEND.values())
    cur = 0
    print(f"gpu N^2 x BH: {len(BACKENDS)} backends, {total} testes no total")
    for backend in BACKENDS:
        print(f"\nBackend: {backend}")
        for N in N_PER_BACKEND[backend]:
            # 10M entra via --merge (probes ja medidos); nao re-executa aqui
            if N >= 10000000 and backend in ("gpu_persist", "bh_gpu"):
                print(f"  N={N} -> via --merge (probe pre-existente)")
                continue
            cur += 1
            steps = get_steps(N, base_steps)
            print(f"  [{cur}/{total}] N={N} steps={steps}...", end=" ", flush=True)
            try:
                r = run_one(backend, N, steps)
                print(f"{r['ms_per_step']:.2f} ms/step ({r['total_time_sec']:.2f}s)")
            except Exception as e:
                r = {"backend": backend, "N": N, "steps": steps,
                     "total_time_sec": None, "ms_per_step": None,
                     "success": False, "error": str(e)}
                print(f"FAILED: {e}")
            results.append(r)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2, default=str)
    print("\nBenchmark completo!")
    return results


def merge_points(results, merge_file):
    """Acrescenta pontos medidos em outro JSON (ex: probes), sem duplicar."""
    extra = json.loads(Path(merge_file).read_text())
    have = {(r["backend"], r["N"]) for r in results if r.get("success")}
    n = 0
    for r in extra:
        if r.get("backend") not in N_PER_BACKEND:
            continue  # so backends deste comparativo
        if r.get("success") and (r["backend"], r["N"]) not in have:
            results.append(r)
            have.add((r["backend"], r["N"]))
            n += 1
    print(f"merge: {n} pontos importados de {merge_file}")
    return results


def _data(results):
    data = {}
    for r in results:
        if r and r.get("success") and r.get("ms_per_step"):
            b = r["backend"]
            data.setdefault(b, {"N": [], "ms": []})
            data[b]["N"].append(r["N"])
            data[b]["ms"].append(r["ms_per_step"])
    for b in data:
        idx = sorted(range(len(data[b]["N"])), key=lambda i: data[b]["N"][i])
        data[b]["N"] = [data[b]["N"][i] for i in idx]
        data[b]["ms"] = [data[b]["ms"][i] for i in idx]
    return data


def _crossover(data, b1="gpu_persist", b2="bh_gpu_full"):
    """N onde b2 passa b1 (interpolacao log-log). None se nao houver."""
    if b1 not in data or b2 not in data:
        return None
    common = sorted(set(data[b1]["N"]) & set(data[b2]["N"]))
    prev, prev_n = None, None
    for n in common:
        t1 = data[b1]["ms"][data[b1]["N"].index(n)]
        t2 = data[b2]["ms"][data[b2]["N"].index(n)]
        d = np.log(t1) - np.log(t2)
        if prev is not None and prev != 0 and d != 0 and (prev > 0) != (d > 0):
            # troca de sinal entre prev_n e n: interpola em log-log
            n0 = prev_n
            d0 = prev
            f = abs(d0) / (abs(d0) + abs(d))  # fracao logaritmica entre n0 e n
            return int(round(np.exp(np.log(n0) + f * (np.log(n) - np.log(n0)))))
        prev, prev_n = d, n
    return None


def plot_results(results, tag=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _data(results)
    if not data:
        print("sem dados validos para plotar")
        return
    suffix = f"_{tag}" if tag else f"_{TIMESTAMP}"
    present = sorted(data)
    # Titulo construido a partir do que FOI medido (nunca hardcodado).
    title_bits = []
    if "gpu_persist" in data:
        title_bits.append("N² direto (gpu_persist)")
    bh_bits = [b for b in ("bh_gpu", "bh_gpu_full") if b in data]
    if bh_bits:
        title_bits.append("Barnes-Hut " + " + ".join(bh_bits))
    title = " vs ".join(title_bits) if title_bits else ", ".join(present)
    x_cross = _crossover(data)

    def _draw(ax, log):
        for b in present:
            if log:
                ax.loglog(data[b]["N"], data[b]["ms"], "o-",
                          label=LABELS.get(b, b), markersize=6,
                          color=BACKEND_COLORS.get(b))
            else:
                ax.plot(data[b]["N"], data[b]["ms"], "o-",
                        label=LABELS.get(b, b), markersize=6,
                        color=BACKEND_COLORS.get(b))
        if log:
            # referencias de complexidade ancoradas em pontos medidos.
            # Regra unica: ancora = maior N <= 200k da curva modelada (primeiro
            # ponto ja fora do custo fixo; projetar dali testa o modelo).
            if "gpu_persist" in data:
                cands = [n for n in data["gpu_persist"]["N"] if n <= 200000]
                if cands:
                    n0 = max(cands)
                    t0 = data["gpu_persist"]["ms"][data["gpu_persist"]["N"].index(n0)]
                    xs = np.logspace(np.log10(n0), np.log10(max(data["gpu_persist"]["N"])), 60)
                    ax.loglog(xs, t0 * (xs / n0) ** 2, "--", color="gray", lw=1,
                              label=f"ref O(N²), ancora N={n0:,}".replace(",", "."))
            for b in ("bh_gpu_full", "bh_gpu"):
                if b in data:
                    cands = [n for n in data[b]["N"] if n <= 200000]
                    if cands:
                        n0 = max(cands)
                        t0 = data[b]["ms"][data[b]["N"].index(n0)]
                        xs = np.logspace(np.log10(n0), np.log10(max(data[b]["N"])), 60)
                        ax.loglog(xs, t0 * (xs / n0) * (np.log(xs) / np.log(n0)), ":",
                                  color="gray", lw=1,
                                  label=f"ref O(N log N) [{b}], ancora N={n0:,}".replace(",", "."))
                        break
            if x_cross:
                import numpy as _np
                ax.axvline(x_cross, color="k", ls="--", lw=1, alpha=0.6)
                ymin, ymax = ax.get_ylim()
                ypos = _np.exp(0.80 * _np.log(ymin) + 0.20 * _np.log(ymax))
                ax.text(x_cross * 1.12, ypos,
                        f"crossover N~{x_cross:,}".replace(",", "."),
                        rotation=90, va="bottom", ha="center", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
        ax.set_xlabel("N (numero de corpos)")
        ax.set_ylabel("Tempo por step (ms)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both" if log else "major", ls="-", alpha=0.3)

    # linear-linear
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw(ax, log=False)
    ax.set_title(title + " — escala linear")
    plt.tight_layout()
    f1 = PLOTS_DIR / f"gpu_vs_bh_linear{suffix}.png"
    plt.savefig(f1, dpi=150)
    plt.close()
    print(f"Plot salvo: {f1}")

    # log-log
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw(ax, log=True)
    ax.set_title(title + " — log-log")
    plt.tight_layout()
    f2 = PLOTS_DIR / f"gpu_vs_bh_loglog{suffix}.png"
    plt.savefig(f2, dpi=150)
    plt.close()
    print(f"Plot salvo: {f2}")

    # tabela resumo dinamica
    cols = [b for b in ("gpu_persist", "bh_gpu", "bh_gpu_full") if b in data]
    print("\n| N | " + " | ".join(f"{b} (ms/step)" for b in cols) + " | vencedor |")
    print("|---|" + "|".join(["---"] * len(cols)) + "|----------|")
    all_n = sorted({n for b in data.values() for n in b["N"]})
    for n in all_n:
        vals = {b: data[b]["ms"][data[b]["N"].index(n)] for b in cols if n in data[b]["N"]}
        cells = [f"{vals[b]:.2f}" if b in vals else "—" for b in cols]
        win = min(vals, key=vals.get) if len(vals) > 1 else (next(iter(vals)) if vals else "—")
        print(f"| {n} | " + " | ".join(cells) + f" | {win} |")
    if x_cross:
        print(f"\ncrossover gpu_persist x bh_gpu_full: N~{x_cross:,}".replace(",", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--plot-only", default=None)
    ap.add_argument("--merge", default=None, help="JSON extra com pontos ja medidos (ex: probe_limits.json)")
    ap.add_argument("--tag", default=None, help="sufixo dos PNGs no modo --plot-only")
    args = ap.parse_args()
    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        if args.merge:
            results = merge_points(results, args.merge)
        plot_results(results, tag=args.tag)
        return
    try:
        results = run_all(args.steps)
    except KeyboardInterrupt:
        print("\nInterrompido! Gerando plots parciais...")
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        except Exception:
            results = []
    if args.merge:
        results = merge_points(results, args.merge)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados: {RESULTS_FILE}")
    plot_results(results)


if __name__ == "__main__":
    main()
