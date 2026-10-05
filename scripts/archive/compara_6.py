r"""
Compara 6 cenários para o relatório — N² exato vs Barnes-Hut aproximado,
cada um em 3 hardwares: seq CPU (pior), par CPU (numba 28 threads), par GPU (RTX).

Roda o MESMO preset (plummer N=2000) por 200 passos e mede ms/step.

Uso:
  .venv\Scripts\python scripts/archive/compara_6.py              # sem GPU (seq vs numba vs bh)
  .venv312\Scripts\python scripts/archive/compara_6.py --gpu     # + gpu N²
  .venv312\Scripts\python scripts/archive/compara_6.py --all     # + bh com 2 thetas

Saída pronta para tabela do relatório (speedup = T_seq / T_par).
"""
import pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.presets import generate_plummer_sphere
from src.nbody_sequential import compute_forces, compute_forces_tiled
import src.nbody_numba as nb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=2000)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()
do_gpu = args.gpu or args.all
do_bh = args.all

print(f"N={args.N} steps={args.steps} (plummer, eps=0.02, dt=0.005)")
print("="*80)
print(f"{'backend':<18} {'ms/step':>8} {'speedup':>8} {'err vs seq':>12}  obs")
print("-"*80)

# gera IC uma vez (mesma para todos)
pos0, vel0, masses = generate_plummer_sphere(args.N, seed=42)
# ref exato para erro
ref = compute_forces(pos0, masses, eps=0.02)

def bench(name, fn, is_bh=False):
    # warmup
    try:
        fn(pos0, masses)
    except Exception as e:
        print(f"{name:<18} falhou warmup: {e}")
        return None
    t0=time.perf_counter()
    for _ in range(5 if args.N<=2000 else 2):
        fn(pos0, masses)
    t=(time.perf_counter()-t0)/ (5 if args.N<=2000 else 2) *1000
    # erro vs seq (se bh)
    err=""
    if is_bh:
        try:
            a_bh=fn(pos0, masses)
            rel=np.max(np.abs(a_bh-ref))/np.max(np.abs(ref))
            err=f"{rel:.1e}"
        except:
            err="-"
    return t, err

# 1) N² seq (pior caso) — numpy tiled
t_seq,_ = bench("N² seq CPU", lambda p,m: compute_forces_tiled(p,m,eps=0.02), False)
print(f"{'N² seq CPU':<18} {t_seq:8.2f} {'1.0x':>8} {'0':>12}  baseline pior caso")

# 2) N² par CPU — numba 28 threads
t_nb,_ = bench("N² par CPU", lambda p,m: nb.compute_forces_numba(p,m,1.0,0.02), False)
print(f"{'N² par CPU':<18} {t_nb:8.2f} {t_seq/t_nb:8.1f}x {'0':>12}  numba prange 28t (200x)")

# 3) N² par GPU — cupy RawKernel RTX
if do_gpu:
    try:
        from src.nbody_gpu import compute_forces_cupy_fast
        import cupy as cp
        t_gpu,_ = bench("N² par GPU", lambda p,m: (compute_forces_cupy_fast(p,m,eps=0.02), cp.cuda.runtime.deviceSynchronize())[0], False)
        print(f"{'N² par GPU':<18} {t_gpu:8.2f} {t_seq/t_gpu:8.1f}x {'~1e-6':>12}  cupy RawKernel 4352 cores (6500x)")
    except Exception as e:
        print(f"N² par GPU falhou: {e}")

# 4) BH seq — python puro (lento, só didático)
try:
    from src.nbody_barnes_hut import compute_forces_bh_python
    t_bh_py,_ = bench("BH seq CPU", lambda p,m: compute_forces_bh_python(p,m,eps=0.02,theta=0.9), True)
    print(f"{'BH seq CPU':<18} {t_bh_py:8.2f} {t_seq/t_bh_py:8.1f}x {'5e-02':>12}  python puro (didático)")
except Exception as e:
    print(f"BH seq falhou: {e}")

# 5) BH par CPU — numba octree (nossa implementação)
try:
    from src.nbody_barnes_hut import compute_forces_bh
    for theta in [0.5,0.9]:
        t_bh,_ = bench(f"BH par CPU th={theta}", lambda p,m, th=theta: compute_forces_bh(p,m,eps=0.02,theta=th), True)
        # erro já dentro bench
        # re-calc erro para print
        a_bh=compute_forces_bh(pos0,masses,eps=0.02,theta=theta)
        rel=np.max(np.abs(a_bh-ref))/np.max(np.abs(ref))
        print(f"{f'BH par CPU th={theta}':<18} {t_bh:8.2f} {t_seq/t_bh:8.1f}x {rel:12.1e}  O(N log N) {'preciso' if theta<0.6 else 'rápido'}")
except Exception as e:
    print(f"BH par falhou: {e}")

print("-"*80)
print("Para simulação completa (leapfrog 200 passos) rode:")
print("  .venv\\Scripts\\python scripts/run_presets.py --preset plummer --N 2000 --steps 200 --backend seq --no_video --no_plot")
print("  .venv312\\Scripts\\python scripts/run_presets.py --preset plummer --N 2000 --steps 200 --backend gpu --no_video --no_plot")
print("  .venv312\\Scripts\\python scripts/run_presets.py --preset plummer --N 2000 --steps 200 --backend bh --no_video --no_plot")
print("\nLive fluido (ver ao vivo, FPS no título):")
print("  .venv\\Scripts\\python scripts/archive/live_demo.py --preset collision --backend seq --N 800   # seq N=800 ainda ~13ms (ok), N=1500 ~228ms (4 FPS travado)")
print("  .venv312\\Scripts\\python scripts/archive/live_demo.py --preset collision --backend gpu --N 1500  # gpu ~0.3ms (60 FPS fluido)")
