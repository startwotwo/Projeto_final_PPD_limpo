from __future__ import annotations
import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]

def torch_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def cupy_available() -> bool:
    try:
        import cupy
        return cupy.cuda.is_available()
    except Exception:
        return False

def numba_cuda_available() -> bool:
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False

# ------------------------------------------------------------------
# PyTorch tiled
# ------------------------------------------------------------------

def compute_forces_torch(pos: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02, tile: int = 1024) -> Vec3:
    import torch
    assert torch.cuda.is_available(), "torch cuda não disponível"
    device = torch.device("cuda")
    pos_t = torch.from_numpy(pos.astype(np.float32)).to(device)
    masses_t = torch.from_numpy(masses.astype(np.float32)).to(device)
    N = pos.shape[0]
    accel_t = torch.zeros((N, 3), dtype=torch.float32, device=device)
    eps2 = float(eps * eps)
    for j0 in range(0, N, tile):
        j1 = min(j0 + tile, N)
        pos_tile = pos_t[j0:j1]
        masses_tile = masses_t[j0:j1]
        r_vec = pos_tile.unsqueeze(0) - pos_t.unsqueeze(1)  # (N, tile, 3)
        r2 = (r_vec * r_vec).sum(dim=2) + eps2
        for i in range(max(0, j0), min(N, j1)):
            r2[i, i - j0] = float("inf")
        inv_r = torch.rsqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r
        coeff = masses_tile.unsqueeze(0) * inv_r3
        accel_t += G * (coeff.unsqueeze(2) * r_vec).sum(dim=1)
    return accel_t.cpu().numpy().astype(np.float64)


def total_energy_torch(pos: Vec3, vel: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02, tile: int = 1024) -> tuple[float, float, float]:
    import torch
    device = torch.device("cuda")
    pos_t = torch.from_numpy(pos.astype(np.float32)).to(device)
    vel_t = torch.from_numpy(vel.astype(np.float32)).to(device)
    masses_t = torch.from_numpy(masses.astype(np.float32)).to(device)
    ekin = 0.5 * (masses_t * (vel_t * vel_t).sum(dim=1)).sum().item()
    N = pos.shape[0]
    epot = 0.0
    eps2 = eps * eps
    for j0 in range(0, N, tile):
        j1 = min(j0 + tile, N)
        pos_tile = pos_t[j0:j1]
        masses_tile = masses_t[j0:j1]
        r_vec = pos_tile.unsqueeze(0) - pos_t.unsqueeze(1)
        r = torch.sqrt((r_vec * r_vec).sum(dim=2) + eps2)
        for i in range(max(0, j0), min(N, j1)):
            r[i, i - j0] = float("inf")
        m_mat = masses_t.unsqueeze(1) * masses_tile.unsqueeze(0)
        pot_tile = G * m_mat / r
        epot += pot_tile.sum().item()
    epot = -0.5 * epot
    return float(ekin), float(epot), float(ekin + epot)

# ------------------------------------------------------------------
# CuPy tiled
# ------------------------------------------------------------------

def compute_forces_cupy(pos: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02, tile: int = 1024) -> Vec3:
    import cupy as cp
    pos_c = cp.asarray(pos, dtype=cp.float32)
    masses_c = cp.asarray(masses, dtype=cp.float32)
    N = pos.shape[0]
    accel_c = cp.zeros((N, 3), dtype=cp.float32)
    eps2 = eps * eps
    for j0 in range(0, N, tile):
        j1 = min(j0 + tile, N)
        pos_tile = pos_c[j0:j1]
        r_vec = pos_tile[None, :, :] - pos_c[:, None, :]
        r2 = cp.sum(r_vec * r_vec, axis=2) + eps2
        for i in range(max(0, j0), min(N, j1)):
            r2[i, i - j0] = cp.inf
        inv_r = 1.0 / cp.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r
        coeff = masses_c[j0:j1][None, :] * inv_r3
        accel_c += G * cp.sum(coeff[:, :, None] * r_vec, axis=1)
    return cp.asnumpy(accel_c).astype(np.float64)

# ------------------------------------------------------------------
# CuPy RawKernel — 1 thread por partícula, loop j em registradores
# ------------------------------------------------------------------

_cupy_kernel_cache = {}

def _get_cupy_kernel():
    import cupy as cp
    if "nbody" in _cupy_kernel_cache:
        return _cupy_kernel_cache["nbody"]
    code = r'''
    extern "C" __global__
    void nbody_forces(const float* pos, const float* masses, float* accel, int N, float G, float eps2) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;
        float xi = pos[3*i];
        float yi = pos[3*i+1];
        float zi = pos[3*i+2];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            float dx = pos[3*j]   - xi;
            float dy = pos[3*j+1] - yi;
            float dz = pos[3*j+2] - zi;
            float r2 = dx*dx + dy*dy + dz*dz + eps2;
            float inv_r = rsqrtf(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            float s = G * masses[j] * inv_r3;
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }
        accel[3*i]   = ax;
        accel[3*i+1] = ay;
        accel[3*i+2] = az;
    }
    '''
    kern = cp.RawKernel(code, 'nbody_forces')
    _cupy_kernel_cache["nbody"] = kern
    return kern


def compute_forces_cupy_fast(pos: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02) -> Vec3:
    """Versão RawKernel — mais rápida para N grande."""
    import cupy as cp
    N = pos.shape[0]
    pos_f32 = np.ascontiguousarray(pos.astype(np.float32))
    masses_f32 = np.ascontiguousarray(masses.astype(np.float32))
    pos_c = cp.asarray(pos_f32)
    masses_c = cp.asarray(masses_f32)
    pos_flat = pos_c.ravel()
    accel_flat = cp.zeros(N * 3, dtype=cp.float32)
    kern = _get_cupy_kernel()
    threads = 256
    blocks = (N + threads - 1) // threads
    kern((blocks,), (threads,), (pos_flat, masses_c, accel_flat, N, np.float32(G), np.float32(eps * eps)))
    accel = cp.asnumpy(accel_flat).reshape(N, 3)
    return accel.astype(np.float64)


# ------------------------------------------------------------------
# CuPy RawKernel com buffers persistentes — reusa alocacoes device
# ------------------------------------------------------------------

_persist_cache: dict = {"N": 0, "pos_flat": None, "masses_c": None, "accel_flat": None, "masses_id": None}


def clear_cupy_persistent_cache() -> None:
    """Libera buffers device persistentes (troca de N ou fim do benchmark)."""
    global _persist_cache
    _persist_cache = {"N": 0, "pos_flat": None, "masses_c": None, "accel_flat": None, "masses_id": None}


def compute_forces_cupy_fast_persistent(pos: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02) -> Vec3:
    """Mesmo kernel de compute_forces_cupy_fast, mas sem realocar device a cada step.

    Reusa pos_flat/masses_c/accel_flat entre steps (só H2D via set()).
    masses só é reenviado se N ou id(masses) mudar (massa é constante na simulação).
    """
    import cupy as cp
    global _persist_cache
    N = pos.shape[0]
    if _persist_cache["pos_flat"] is None or _persist_cache["N"] != N:
        _persist_cache["pos_flat"] = cp.empty(N * 3, dtype=cp.float32)
        _persist_cache["masses_c"] = cp.empty(N, dtype=cp.float32)
        _persist_cache["accel_flat"] = cp.empty(N * 3, dtype=cp.float32)
        _persist_cache["N"] = N
        _persist_cache["masses_id"] = None
    pos_flat = _persist_cache["pos_flat"]
    masses_c = _persist_cache["masses_c"]
    accel_flat = _persist_cache["accel_flat"]
    if _persist_cache["masses_id"] != id(masses):
        masses_c.set(np.ascontiguousarray(masses.astype(np.float32)))
        _persist_cache["masses_id"] = id(masses)
    pos_flat.set(np.ascontiguousarray(pos.astype(np.float32)).ravel())
    kern = _get_cupy_kernel()
    threads = 256
    blocks = (N + threads - 1) // threads
    kern((blocks,), (threads,), (pos_flat, masses_c, accel_flat, N, np.float32(G), np.float32(eps * eps)))
    accel = cp.asnumpy(accel_flat).reshape(N, 3)
    return accel.astype(np.float64)


def total_energy_cupy(pos: Vec3, vel: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02, tile: int = 1024) -> tuple[float, float, float]:
    import cupy as cp
    pos_c = cp.asarray(pos, dtype=cp.float32)
    vel_c = cp.asarray(vel, dtype=cp.float32)
    masses_c = cp.asarray(masses, dtype=cp.float32)
    ekin = 0.5 * cp.sum(masses_c * cp.sum(vel_c * vel_c, axis=1)).item()
    N = pos.shape[0]
    epot = 0.0
    eps2 = eps * eps
    for j0 in range(0, N, tile):
        j1 = min(j0 + tile, N)
        pos_tile = pos_c[j0:j1]
        r_vec = pos_tile[None, :, :] - pos_c[:, None, :]
        r = cp.sqrt(cp.sum(r_vec * r_vec, axis=2) + eps2)
        for i in range(max(0, j0), min(N, j1)):
            r[i, i - j0] = cp.inf
        m_mat = masses_c[:, None] * cp.asarray(masses[j0:j1], dtype=cp.float32)[None, :]
        pot_tile = G * m_mat / r
        epot += cp.sum(pot_tile).item()
    epot = -0.5 * epot
    return float(ekin), float(epot), float(ekin + epot)

# ------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, "src")
    from presets import generate_spiral_galaxy
    for N in [1000, 2000, 5000]:
        pos, vel, masses, _ = generate_spiral_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=42)
        print(f"\nN={N}")
        if torch_available():
            import torch
            compute_forces_torch(pos, masses)
            torch.cuda.synchronize()
            t0=time.perf_counter()
            for _ in range(20):
                compute_forces_torch(pos, masses)
                torch.cuda.synchronize()
            t=(time.perf_counter()-t0)/20*1000
            print(f"  torch {t:.2f} ms")
        if cupy_available():
            import cupy as cp
            compute_forces_cupy(pos, masses)
            cp.cuda.runtime.deviceSynchronize()
            t0=time.perf_counter()
            for _ in range(20):
                compute_forces_cupy(pos, masses)
                cp.cuda.runtime.deviceSynchronize()
            t=(time.perf_counter()-t0)/20*1000
            print(f"  cupy  {t:.2f} ms")
