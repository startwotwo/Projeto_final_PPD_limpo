r"""
Demo unico de ganho de desempenho — roda todos os paradigmas no mesmo preset
e imprime tabela pronta para o relatorio (speedup, eficiencia, tempo).

Uso:
  .venv\Scripts\python scripts/archive/demo_desempenho.py              # CPU (seq vs numba)
  .venv312\Scripts\python scripts/archive/demo_desempenho.py --gpu     # + GPU
  .venv312\Scripts\python scripts/archive/demo_desempenho.py --all     # + Barnes-Hut

Gera tambem data/demo_speedup.png (loglog tempo vs N)
"""
import pathlib, sys, time, argparse
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.presets import generate_plummer_sphere, generate_spiral_galaxy
from src.nbody_sequential import compute_forces_tiled
import src.nbody_numba as nb

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="inclui GPU (precisa .venv312)")
parser.add_argument("--all", action="store_true", help="inclui GPU + Barnes-Hut")
parser.add_argument("--N", type=int, default=2000, help="N para simulação longa")
args = parser.parse_args()
do_gpu = args.gpu or args.all
do_bh = args.all

# ---------- Parte 1: microbenchmark compute_forces (ms por chamada) ----------
print("="*70)
print("1) Microbenchmark compute_forces (1 chamada, média de 20)")
print(f"   Numba threads: {nb.info()['threads']}  ({nb.info()['threading_layer']})  CPU 28 log / 20 fis")
print("="*70)

Ns = [500, 1000, 2000, 5000]
if do_gpu:
    Ns = [500, 1000, 2000, 5000, 10000]
results = {k: {} for k in ["seq", "numba", "gpu", "bh"]}

for N in Ns:
    Nd, Nb_ = int(N*0.7), N-int(N*0.7)
    pos, _, masses, _ = generate_spiral_galaxy(N_disk=Nd, N_bulge=Nb_, seed=42)
    # warmup
    nb.compute_forces_numba(pos, masses, 1.0, 0.05)
    compute_forces_tiled(pos, masses, 1.0, 0.05)
    if do_gpu:
        try:
            from src.nbody_gpu import compute_forces_cupy_fast
            compute_forces_cupy_fast(pos, masses, 1.0, 0.05)
        except Exception:
            pass

    # seq: menos repetições para N grande (senão 6s*5=30s)
    reps_seq = 5 if N <= 2000 else 2
    t0=time.perf_counter()
    for _ in range(reps_seq):
        compute_forces_tiled(pos, masses, 1.0, 0.05)
    t_seq=(time.perf_counter()-t0)/reps_seq*1000
    results["seq"][N]=t_seq

    t0=time.perf_counter()
    for _ in range(20):
        nb.compute_forces_numba(pos, masses, 1.0, 0.05)
    t_nb=(time.perf_counter()-t0)/20*1000
    results["numba"][N]=t_nb

    if do_gpu:
        try:
            from src.nbody_gpu import compute_forces_cupy_fast
            import cupy as cp
            t0=time.perf_counter()
            for _ in range(20):
                compute_forces_cupy_fast(pos, masses, 1.0, 0.05)
                cp.cuda.runtime.deviceSynchronize()
            t_gpu=(time.perf_counter()-t0)/20*1000
            results["gpu"][N]=t_gpu
        except Exception as e:
            print(f"GPU falhou N={N}: {e}")

    if do_bh:
        try:
            from src.nbody_barnes_hut import compute_forces_bh
            reps_bh = 3 if N <= 2000 else 1
            t0=time.perf_counter()
            for _ in range(reps_bh):
                compute_forces_bh(pos, masses, eps=0.02, theta=0.9)
            t_bh=(time.perf_counter()-t0)/reps_bh*1000
            results["bh"][N]=t_bh
        except Exception as e:
            print(f"BH falhou N={N}: {e}")

    # imprime linha
    line = f"N={N:5d}  seq {t_seq:7.2f} ms  numba {t_nb:6.2f} ms  speedup {t_seq/max(t_nb,1e-9):5.1f}x eff {t_seq/max(t_nb,1e-9)/28*100:4.0f}%"
    if do_gpu and N in results["gpu"]:
        line+=f"  gpu {results['gpu'][N]:6.2f} ms  gpu vs seq {t_seq/results['gpu'][N]:5.0f}x  gpu vs numba {t_nb/results['gpu'][N]:4.1f}x"
    if do_bh and N in results["bh"]:
        line+=f"  bh {results['bh'][N]:6.2f} ms"
    print(line)

# ---------- Parte 2: simulação longa completa (ms/step) ----------
print("\n"+"="*70)
print(f"2) Simulação longa completa (300 steps, N={args.N}, dt=0.005)")
print("   Mede tempo total incluindo leapfrog + energia (real do preset)")
print("="*70)
from src.presets import generate_galaxy_collision
from src.presets_runner_fast import run_fast
from src.presets_runner import run_custom_simulation

pos, vel, masses, labels = generate_galaxy_collision(N_disk1=400, N_bulge1=150, N_disk2=400, N_bulge2=150, separation=6, impact_param=1.0, rel_vel_factor=0.45, seed=42)
# ajusta N se pediu maior
if args.N != 1100:
    # usa plummer para N arbitrário
    pos, vel, masses = generate_plummer_sphere(args.N, seed=42)
    labels=None

backends = [("seq", lambda: run_custom_simulation(pos, vel, masses, steps=300, dt=0.005, save_trajectory=False, verbose=False))]
backends.append(("numba", lambda: run_fast(pos, vel, masses, steps=300, dt=0.005, save_trajectory=False, verbose=False, backend="numba")))
if do_gpu:
    backends.append(("gpu", lambda: run_fast(pos, vel, masses, steps=300, dt=0.005, save_trajectory=False, verbose=False, backend="gpu")))
if do_bh:
    backends.append(("bh0.9", lambda: run_fast(pos, vel, masses, steps=300, dt=0.005, save_trajectory=False, verbose=False, backend="bh0.9")))

times = {}
for name, fn in backends:
    t0=time.perf_counter()
    fn()
    elapsed=time.perf_counter()-t0
    times[name]=elapsed
    print(f"{name:6s} 300 steps N={len(masses):4d}  {elapsed:5.2f}s  {elapsed/300*1000:5.2f} ms/step", end="")
    if name!="seq":
        print(f"  speedup vs seq {times['seq']/elapsed:5.1f}x  eff {times['seq']/elapsed/28*100:4.0f}%", end="")
    print()

print("\nFormulas relatorio:")
print("  speedup = T_seq / T_par")
print("  eficiencia = speedup / P  (P=28 threads, P=4352 CUDA cores para GPU)")
print("  Lei de Amdahl: speedup_max = 1 / ((1-f)+f/P)  onde f ~0.99 (forcas sao 99% do tempo)")

# ---------- Gráfico ----------
try:
    fig, ax = plt.subplots(figsize=(7,5))
    for k, vals in results.items():
        if not vals: continue
        Ns_sorted=sorted(vals)
        ax.loglog(Ns_sorted, [vals[n] for n in Ns_sorted], marker="o", label=k)
    # referência O(N²)
    ref_N=np.array(sorted(results["seq"].keys()))
    ref_t=np.array([results["seq"][n] for n in ref_N])
    # normaliza em N=500
    ref_line=ref_t[0]*(ref_N/ref_N[0])**2
    ax.loglog(ref_N, ref_line, "--", alpha=0.4, label="O(N²) ref")
    # BH O(N log N) ref
    if do_bh and results["bh"]:
        bh_N=np.array(sorted(results["bh"].keys()))
        # O(N log N) ~ N log N
        bh_ref=results["bh"][bh_N[0]]*(bh_N*np.log(bh_N))/(bh_N[0]*np.log(bh_N[0]))
        ax.loglog(bh_N, bh_ref, ":", alpha=0.4, label="O(N log N) ref")
    ax.set_xlabel("N"); ax.set_ylabel("ms por compute_forces")
    ax.set_title("Escalabilidade N-Body — seq vs numba vs gpu vs bh")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/demo_speedup.png", dpi=150)
    print("\nGráfico salvo: data/demo_speedup.png (loglog, inclinação 2 = O(N²), 1 = O(N log N))")
except Exception as e:
    print(f"Gráfico falhou: {e}")
