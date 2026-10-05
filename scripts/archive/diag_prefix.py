#!/usr/bin/env python
"""Diagnostico: cronometra a secao de prefixos em 10M, op por op."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import cupy as cp
import src.presets as presets
from src.nbody_bh_gpu_build import _get_morton_kernel, _get_prefix_kernels

N = 10000000
pos, vel, masses, _ = presets.generate_spiral_galaxy(
    N_disk=int(N * 0.7), N_bulge=N - int(N * 0.7), seed=42)
pos_d = cp.asarray(pos.astype(np.float32))
masses_d = cp.asarray(masses.astype(np.float32))
codes_d = cp.empty(N, dtype=cp.int32)
S = cp.cuda.Stream.null


def tick(msg, t0):
    S.synchronize()
    t1 = time.perf_counter()
    print(f"{msg}: {(t1 - t0) * 1000:.1f} ms", flush=True)
    return t1


t0 = time.perf_counter()
pr = pos_d.reshape(N, 3)
mn = cp.min(pr, axis=0); mx = cp.max(pr, axis=0)
t0 = tick("bbox", t0)
mn_h = cp.asnumpy(mn); mx_h = cp.asnumpy(mx)
span = float(np.max(mx_h - mn_h)); Sc = span * 1.01 + 1e-9
mk = _get_morton_kernel()
mk(((N + 255) // 256,), (256,), (pos_d, codes_d, N,
    np.float32(mn_h[0]), np.float32(mn_h[1]), np.float32(mn_h[2]), np.float32(1024.0 / Sc)))
t0 = tick("morton", t0)
order = cp.argsort(codes_d)
sc = codes_d[order]
t0 = tick("argsort", t0)
d = cp.ones(N, dtype=cp.bool_)
d[1:] = sc[1:] != sc[:-1]
nleaf = int(cp.count_nonzero(d))
t0 = tick(f"unique nleaf={nleaf}", t0)
leaf_start = cp.where(d)[0]
uq = sc[d]
t0 = tick("where+gather-uq", t0)
mark_k, compact_k = _get_prefix_kernels()
mark = cp.empty(nleaf * 9, dtype=cp.int32)
t0 = tick("alloc mark 360MB", t0)
mark_k(((nleaf + 255) // 256,), (256,), (uq, mark, np.int32(nleaf)))
t0 = tick("mark kernel", t0)
mark2d = mark.reshape(nleaf, 9)
cp.cumsum(mark2d, axis=0, out=mark2d)
t0 = tick("cumsum axis0 inplace", t0)
cnt_h = cp.asnumpy(mark2d[-1])
t0 = tick(f"D2H counts {cnt_h}", t0)
print("OK - sem travamento")
