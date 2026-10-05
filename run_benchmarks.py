#!/usr/bin/env python
"""
Benchmark Suite for N-Body Simulation Project
=============================================
Compara backends em varios N, cada um ate seu teto viavel (medido em
scripts/probe_limits.py; tetos escolhidos para nenhum teste passar de ~4 min):

  seq:         20k   (~95 s/step em 20k; 50k levaria ~10 min/step -> fora)
  numba:        100k  (~2.5 s/step)
  gpu:          500k  (~1.0 s/step)
  gpu_persist:  2M    (~14.6 s/step; 10M = 325 s/step -> so como probe isolado)
  bh:           200k  (~1.0 s/step)
  bh_par:       500k  (~0.65 s/step)
  bh_morton:    500k  (~0.65 s/step)
  bh_gpu:       5M    (~4.8 s/step; 10M = 23.5 s/step; 20M estoura VRAM+RAM)
  bh_gpu_full:  10M   (~0.68 s/step; 20M so com swap PCIe -> ponto avulso)
  mpi_manual:   20k   (rede; 50k falha se o worker remoto estiver fora)

Metodologia de tempo (rigor): leapfrog manual cronometrado, com UMA chamada de
warmup (compilacao Numba/CUDA) FORA do cronometro e SEM computo de energia no
laco (energia tem custo O(N^2) proprio e contaminaria o ms/step; conservacao
de energia e medida separadamente, ver README §13).
Metrica: ms/step. Preset: galaxy (seed 42).
"""

import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.presets as presets
import src.presets_runner_fast as runner

# Configuration
PRESET = "galaxy"
DT = 0.005
EPS = 0.05
SEED = 42

# N por backend: do comparavel (500) ao teto viavel de cada um.
N_PER_BACKEND = {
    "seq":         [500, 1000, 5000, 10000, 20000],
    "numba":       [500, 1000, 5000, 10000, 20000, 50000, 100000],
    "gpu":         [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "gpu_persist": [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000],
    "bh":          [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000],
    "bh_par":      [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "bh_morton":   [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    "bh_gpu":      [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000],
    "bh_gpu_full": [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000],
    "mpi_manual":  [500, 1000, 5000, 10000, 20000, 50000],
}

BACKENDS = list(N_PER_BACKEND.keys())
GPU_BACKENDS = ["gpu", "gpu_persist", "bh_gpu", "bh_gpu_full"]

# Cor fixa por backend (padrao em TODOS os plots do trabalho).
BACKEND_COLORS = {
    "seq":         "#000000",
    "numba":       "#1f77b4",
    "gpu":         "#2ca02c",
    "gpu_persist": "#7dd87d",
    "bh":          "#d62728",
    "bh_par":      "#ff7f0e",
    "bh_morton":   "#9467bd",
    "bh_gpu":      "#17becf",
    "bh_gpu_full": "#e377c2",
    "mpi_manual":  "#8c564b",
}

MPI_MANUAL_HOSTS_FILE = Path(__file__).parent / "hosts_manual.json"

# Output paths
OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def get_steps(backend, N):
    """Steps por teste: menos steps onde o step e caro (ms/step comparavel)."""
    if backend == "seq":
        if N >= 20000:
            return 2
        if N >= 10000:
            return 5
        return 100
    if backend == "numba":
        return 10 if N >= 100000 else 100
    if backend in ("gpu", "gpu_persist"):
        if N >= 1000000:
            return 3
        if N >= 100000:
            return 10
        return 100
    if backend in ("bh", "bh_par", "bh_morton"):
        return 10 if N >= 200000 else 100
    if backend == "bh_gpu":
        if N >= 1000000:
            return 3
        if N >= 100000:
            return 10
        return 100
    if backend == "bh_gpu_full":
        if N >= 10000000:
            return 3
        if N >= 1000000:
            return 5
        return 100
    if backend == "mpi_manual":
        return 20
    return 100


def generate_preset(preset_name, N):
    """Generate initial conditions for a preset."""
    try:
        if preset_name == "galaxy":
            pos, vel, masses, labels = presets.generate_spiral_galaxy(
                N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=42
            )
        elif preset_name == "collision":
            pos, vel, masses, labels = presets.generate_galaxy_collision(seed=42)
        elif preset_name == "plummer":
            pos, vel, masses = presets.generate_plummer_sphere(N, seed=42)
            labels = None
        elif preset_name == "rotplummer":
            pos, vel, masses = presets.generate_rotating_plummer(N, seed=42)
            labels = None
        elif preset_name == "ring":
            pos, vel, masses, labels = presets.generate_ring_galaxy(
                N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=42
            )
        elif preset_name == "triple":
            pos, vel, masses, labels = presets.generate_triple_merger(
                N_per_galaxy=max(100, N//3), seed=42
            )
        elif preset_name == "satellite":
            pos, vel, masses, labels = presets.generate_satellite_infall(
                N_host_disk=int(N*0.7), N_host_bulge=N-int(N*0.7)-200, N_sat=200, seed=42
            )
        elif preset_name == "cold":
            pos, vel, masses = presets.generate_cold_collapse(N, seed=42)
            labels = None
        elif preset_name == "kepler":
            pos, vel, masses, labels = presets.generate_kepler_binary(seed=42)
        elif preset_name == "disk":
            pos, vel, masses = presets.generate_kepler_disk(N, seed=42)
            labels = None
        elif preset_name == "forming":
            pos, vel, masses, labels = presets.generate_forming_galaxy(N=N, seed=42)
        elif preset_name == "solar":
            pos, vel, masses, labels = presets.generate_solar_system(seed=42, with_moons=True)
        elif preset_name == "solar_real":
            pos, vel, masses, labels = presets.generate_solar_system_real(with_moons=True)
        elif preset_name == "solar_completo":
            pos, vel, masses, labels = presets.generate_solar_completo(with_moons=True)
        else:
            pos, vel, masses, labels = presets.generate_spiral_galaxy(
                N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=42
            )
        return pos, vel, masses, labels
    except Exception as e:
        print(f"Error generating {preset_name} with N={N}: {e}")
        return None, None, None, None


def leapfrog_time(cf, pos, vel, masses, steps):
    """Integra `steps` passos leapfrog cronometrados (warmup ja feito fora)."""
    pos_c = pos.copy()
    vel_c = vel.copy()
    accel = cf(pos_c, masses, 1.0, EPS)
    t0 = time.perf_counter()
    for _ in range(steps):
        vel_half = vel_c + accel * (DT * 0.5)
        pos_c = pos_c + vel_half * DT
        accel = cf(pos_c, masses, 1.0, EPS)
        vel_c = vel_half + accel * (DT * 0.5)
    return time.perf_counter() - t0


def run_benchmark(backend, N, steps):
    """Run a single benchmark and return timing metrics."""
    pos, vel, masses, labels = generate_preset(PRESET, N)
    if pos is None:
        return None

    # mpi_manual usa cluster (hosts_manual.json), nao o runner local
    if backend == "mpi_manual":
        try:
            import src.nbody_mpi_manual as mpi_man
        except ImportError:
            import nbody_mpi_manual as mpi_man
        try:
            hosts = json.loads(MPI_MANUAL_HOSTS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            return {"backend": backend, "N": N, "steps": steps,
                    "total_time_sec": None, "ms_per_step": None,
                    "energy_history": None, "success": False,
                    "error": f"hosts_manual.json nao encontrado: {e}"}
        try:
            mpi_man.compute_forces_mpi_manual(pos, masses, G=1.0, eps=EPS, hosts=hosts)  # warmup
            pos_c = pos.copy(); vel_c = vel.copy()
            accel = mpi_man.compute_forces_mpi_manual(pos_c, masses, G=1.0, eps=EPS, hosts=hosts)
            t_start = time.perf_counter()
            for _ in range(steps):
                vel_half = vel_c + accel * (DT * 0.5)
                pos_c = pos_c + vel_half * DT
                accel = mpi_man.compute_forces_mpi_manual(pos_c, masses, G=1.0, eps=EPS, hosts=hosts)
                vel_c = vel_half + accel * (DT * 0.5)
            t_total = time.perf_counter() - t_start
            ms_per_step = (t_total / steps) * 1000
            return {"backend": backend, "N": N, "steps": steps,
                    "total_time_sec": t_total, "ms_per_step": ms_per_step,
                    "energy_history": None, "hosts": hosts,
                    "success": True, "error": None}
        except Exception as e:
            return {"backend": backend, "N": N, "steps": steps,
                    "total_time_sec": None, "ms_per_step": None,
                    "energy_history": None, "success": False, "error": str(e)}

    try:
        cf, _, _ = runner._get_backends(backend)
        cf(pos, masses, 1.0, EPS)  # warmup: compilacao Numba/CUDA fora do cronometro
        t_total = leapfrog_time(cf, pos, vel, masses, steps)
        ms_per_step = (t_total / steps) * 1000
        return {
            "backend": backend,
            "N": N,
            "steps": steps,
            "total_time_sec": t_total,
            "ms_per_step": ms_per_step,
            "energy_history": None,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "backend": backend,
            "N": N,
            "steps": steps,
            "total_time_sec": None,
            "ms_per_step": None,
            "energy_history": None,
            "success": False,
            "error": str(e)
        }


def run_all_benchmarks():
    """Run all benchmark combinations."""
    results = []
    total_tests = sum(len(v) for v in N_PER_BACKEND.values())
    current = 0

    print(f"Starting benchmark suite: {len(BACKENDS)} backends, {total_tests} testes no total")
    print(f"Preset: {PRESET}, dt={DT}, eps={EPS} (steps variam por backend/N; warmup fora do cronometro, sem energia no laco)")
    print("-" * 60)

    for backend in BACKENDS:
        print(f"\nBackend: {backend}")
        for N in N_PER_BACKEND[backend]:
            current += 1
            steps = get_steps(backend, N)
            print(f"  [{current}/{total_tests}] N={N} steps={steps}...", end=" ", flush=True)

            # Skip GPU backends if CUDA not available
            if backend in GPU_BACKENDS:
                try:
                    import cupy
                    cupy.cuda.runtime.getDeviceCount()
                except Exception:
                    print("SKIP (no CUDA)")
                    results.append({
                        "backend": backend, "N": N, "steps": steps,
                        "total_time_sec": None, "ms_per_step": None,
                        "success": False, "error": "CUDA not available"
                    })
                    continue

            result = run_benchmark(backend, N, steps)
            results.append(result)

            if result["success"]:
                print(f"{result['ms_per_step']:.2f} ms/step ({result['total_time_sec']:.2f}s total)")
            else:
                print(f"FAILED: {result['error']}")

            # Save intermediate results
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    return results


def _organize(results):
    """Agrupa {backend: {N, ms_per_step, total_time}} ordenado por N."""
    data_by_backend = {}
    for r in results:
        if r and r.get("success") and r.get("ms_per_step"):
            b = r["backend"]
            if b not in data_by_backend:
                data_by_backend[b] = {"N": [], "ms_per_step": [], "total_time": []}
            data_by_backend[b]["N"].append(r["N"])
            data_by_backend[b]["ms_per_step"].append(r["ms_per_step"])
            data_by_backend[b]["total_time"].append(r["total_time_sec"])
    for b in data_by_backend:
        idx = sorted(range(len(data_by_backend[b]["N"])), key=lambda i: data_by_backend[b]["N"][i])
        for k in ("N", "ms_per_step", "total_time"):
            data_by_backend[b][k] = [data_by_backend[b][k][i] for i in idx]
    return data_by_backend


def _ref_slopes(ax, data):
    """Linhas de referencia O(N^2) e O(N log N) ancoradas em pontos medidos.

    Regra unica: ancora = maior N <= 200k da curva modelada (primeiro ponto
    ja fora do custo fixo; projetar dali testa o modelo). Acima de ~1M a
    curva device cresce MAIS DEVAGAR que N log N (rampa de ocupacao da GPU),
    de modo que a referencia vira limite superior nesse regime.
    """
    import numpy as np
    if "gpu_persist" in data:
        cands = [n for n in data["gpu_persist"]["N"] if n <= 200000]
        if cands:
            n0 = max(cands)
            t0 = data["gpu_persist"]["ms_per_step"][data["gpu_persist"]["N"].index(n0)]
            nmax = max(data["gpu_persist"]["N"])
            xs = np.logspace(np.log10(n0), np.log10(max(nmax, n0 * 1.01)), 50)
            ax.loglog(xs, t0 * (xs / n0) ** 2, "--", color="gray", lw=1,
                      label=f"ref O(N²), ancora N={n0:,}".replace(",", "."))
    bh_like = [b for b in ("bh_gpu_full", "bh_gpu", "bh") if b in data]
    if bh_like:
        b = bh_like[0]
        cands = [n for n in data[b]["N"] if n <= 200000]
        if cands:
            n0 = max(cands)
            t0 = data[b]["ms_per_step"][data[b]["N"].index(n0)]
            nmax = max(data[b]["N"])
            xs = np.logspace(np.log10(n0), np.log10(max(nmax, n0 * 1.01)), 50)
            ax.loglog(xs, t0 * (xs / n0) * (np.log(xs) / np.log(n0)), ":",
                      color="gray", lw=1,
                      label=f"ref O(N log N), ancora N={n0:,}".replace(",", "."))


def generate_comparison_plots(results, tag=None):
    """Gera graficos padronizados: x=N, y=ms/step, uma linha por backend (cor fixa)."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    data_by_backend = _organize(results)
    if not data_by_backend:
        print("sem dados validos para plotar")
        return
    suffix = f"_{tag}" if tag else f"_{TIMESTAMP}"

    def _plot_all(ax, logx=False, logy=False, slopes=False):
        for backend in sorted(data_by_backend):
            data = data_by_backend[backend]
            ax.plot(data["N"], data["ms_per_step"], 'o-',
                    label=backend, markersize=5,
                    color=BACKEND_COLORS.get(backend))
        ax.set_xlabel("N (numero de corpos)")
        ax.set_ylabel("Tempo por step (ms)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both" if (logx or logy) else "major", ls="-", alpha=0.3)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if slopes and logx and logy:
            _ref_slopes(ax, data_by_backend)
            ax.legend(fontsize=8, loc="upper left")

    # 1: linear-linear (todos os backends; N grande domina -> util p/ tetos)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_all(ax, logx=False, logy=False)
    ax.set_title(f"Tempo por step vs N — todos os backends (preset {PRESET}, leapfrog sem energia)")
    plt.tight_layout()
    f1 = PLOTS_DIR / f"benchmark_ms_per_step_linear{suffix}.png"
    plt.savefig(f1, dpi=150)
    plt.close()
    print(f"Plot salvo: {f1}")

    # 2: log-log com inclinacoes de referencia O(N^2)/O(N log N)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_all(ax, logx=True, logy=True, slopes=True)
    ax.set_title(f"Tempo por step vs N, log-log (preset {PRESET})")
    plt.tight_layout()
    f2 = PLOTS_DIR / f"benchmark_ms_per_step_loglog{suffix}.png"
    plt.savefig(f2, dpi=150)
    plt.close()
    print(f"Plot salvo: {f2}")

    # 3: Speedup vs seq (por ms/step), so onde ha N em comum
    if "seq" in data_by_backend:
        fig, ax = plt.subplots(figsize=(10, 6))
        seq_ms = dict(zip(data_by_backend["seq"]["N"], data_by_backend["seq"]["ms_per_step"]))
        for backend in sorted(data_by_backend):
            if backend == "seq":
                continue
            ns, sp = [], []
            for n, t in zip(data_by_backend[backend]["N"], data_by_backend[backend]["ms_per_step"]):
                if n in seq_ms:
                    ns.append(n)
                    sp.append(seq_ms[n] / t)
            if ns:
                ax.plot(ns, sp, 'o-', label=backend, markersize=5,
                        color=BACKEND_COLORS.get(backend))
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("Speedup vs seq (por ms/step)")
        ax.set_title("Speedup vs Sequencial")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
        f3 = PLOTS_DIR / f"benchmark_speedup{suffix}.png"
        plt.savefig(f3, dpi=150)
        plt.close()
        print(f"Plot salvo: {f3}")


def generate_summary_table(results):
    """Generate a markdown summary table (colunas = uniao dos N medidos)."""
    all_n = sorted({r["N"] for r in results if r and r.get("success")})
    lines = []
    lines.append("# Benchmark Results Summary")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Preset:** {PRESET}")
    lines.append(f"**dt:** {DT}, **eps:** {EPS}")
    lines.append("**Metodo:** leapfrog manual, warmup fora do cronometro, sem energia no laco (ms/step puro)")
    lines.append("")
    lines.append("| Backend | " + " | ".join(f"N={n}" for n in all_n) + " |")
    lines.append("|---------|" + "|".join(["-------"] * len(all_n)) + "|")

    # Group by backend
    backend_data = {}
    for r in results:
        if r and r["success"]:
            if r["backend"] not in backend_data:
                backend_data[r["backend"]] = {}
            backend_data[r["backend"]][r["N"]] = r["ms_per_step"]

    for backend in sorted(backend_data.keys()):
        row = [backend]
        for N in all_n:
            if N in backend_data.get(backend, {}):
                row.append(f"{backend_data[backend][N]:.2f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Speedup table
    lines.append("## Speedup vs Sequential")
    lines.append("")
    lines.append("| Backend | " + " | ".join(f"N={n}" for n in all_n) + " |")
    lines.append("|---------|" + "|".join(["-------"] * len(all_n)) + "|")

    if "seq" in backend_data:
        for backend in sorted(backend_data.keys()):
            if backend == "seq":
                continue
            row = [backend]
            for N in all_n:
                if N in backend_data.get(backend, {}) and N in backend_data.get("seq", {}):
                    speedup = backend_data["seq"][N] / backend_data[backend][N]
                    row.append(f"{speedup:.2f}x")
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", default=None, help="gera plots a partir de JSON existente (ex: benchmark_results/xxx.json)")
    ap.add_argument("--tag", default=None, help="sufixo dos PNGs no modo --plot-only")
    args = ap.parse_args()

    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        generate_comparison_plots(results, tag=args.tag)
        print(generate_summary_table(results))
        return

    print("=" * 60)
    print("N-Body Benchmark Suite")
    print("=" * 60)

    try:
        results = run_all_benchmarks()
    except KeyboardInterrupt:
        print("\nInterrompido! Gerando plots com resultados parciais...")
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        except Exception:
            results = []

    # Save final results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Generate plots (mesmo se parcial)
    try:
        generate_comparison_plots(results)
    except Exception as e:
        print(f"Plot generation failed: {e}")

    # Generate summary
    summary = generate_summary_table(results)
    summary_file = OUTPUT_DIR / f"summary_{TIMESTAMP}.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {summary_file}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
