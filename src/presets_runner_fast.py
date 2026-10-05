"""
Runner rapido — usa Numba prange (28 threads OpenMP) por padrao, 150-250x mais rapido que sequencial.
Mantem fisica identica, so troca o kernel de forcas.
Robusto a imports tanto como `src.xxx` quanto `xxx` (Colab).
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np
import numpy.typing as npt

try:
    from src.nbody_sequential import SimConfig, SimResult, G_DEFAULT, EPS_DEFAULT
except ImportError:
    from nbody_sequential import SimConfig, SimResult, G_DEFAULT, EPS_DEFAULT

Vec3 = npt.NDArray[np.float64]

def _import_src(name):
    try:
        return __import__(f"src.{name}", fromlist=[name])
    except ImportError:
        return __import__(name, fromlist=[name])

def _get_backends(backend: str):
    backend = backend.lower()
    if backend in ("bh_mp", "bh_par_build", "bh_parallel"):
        try:
            from src.nbody_bh_parallel_build import compute_forces_bh_mp
        except ImportError:
            from nbody_bh_parallel_build import compute_forces_bh_mp
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        return compute_forces_bh_mp, nb.total_energy_numba, "bh_mp"
    elif backend in ("bh_morton", "bh_sort"):
        try:
            from src.nbody_bh_parallel_build import compute_forces_bh_morton
        except ImportError:
            from nbody_bh_parallel_build import compute_forces_bh_morton
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        return compute_forces_bh_morton, nb.total_energy_numba, "bh_morton"
    elif backend in ("bh_gpu_full", "bh_full_gpu"):
        # build 100% no device (morton + sort + bottom-up em CuPy).
        # correto (mesmo erro do hibrido), mas mais lento ate ~50k
        # (overhead de ~300 micro-ops); tende a vencer acima disso.
        try:
            from src.nbody_bh_gpu_build import compute_forces_bh_gpu_device as _cfd
        except ImportError:
            from nbody_bh_gpu_build import compute_forces_bh_gpu_device as _cfd
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        def cf_full(pos, masses, G=1.0, eps=0.02, theta=0.9):
            return _cfd(pos, masses, G, eps, theta)
        return cf_full, nb.total_energy_numba, "bh_gpu_full"
    elif backend.startswith("bh"):
        try:
            is_par = "par" in backend
            is_gpu = "gpu" in backend or "cuda" in backend
            b = backend.replace("_par", "").replace("par", "").replace("_gpu","").replace("gpu","").replace("_cuda","")
            theta = 0.9
            if len(b) > 2:
                try:
                    theta = float(b[2:])
                except:
                    theta = 0.9
            try:
                import src.nbody_barnes_hut as bh
            except ImportError:
                import nbody_barnes_hut as bh
            if is_gpu:
                try:
                    # tenta build 100% GPU (morton) primeiro
                    from src.nbody_bh_gpu_build import compute_forces_bh_gpu_full as compute_forces_bh_gpu
                except ImportError:
                    try:
                        from nbody_bh_gpu_build import compute_forces_bh_gpu_full as compute_forces_bh_gpu
                    except ImportError:
                        try:
                            from src.nbody_bh_gpu import compute_forces_bh_gpu
                        except ImportError:
                            from nbody_bh_gpu import compute_forces_bh_gpu
                def cf(pos, masses, G=1.0, eps=0.02):
                    return compute_forces_bh_gpu(pos, masses, G, eps, theta=theta)
                tag = f"bh_gpu(theta={theta})"
            elif is_par:
                def cf(pos, masses, G=1.0, eps=0.02):
                    return bh.compute_forces_bh_parallel(pos, masses, G, eps, theta=theta)
                tag = f"bh_par(theta={theta})"
            else:
                def cf(pos, masses, G=1.0, eps=0.02):
                    return bh.compute_forces_bh(pos, masses, G, eps, theta=theta)
                tag = f"bh(theta={theta})"
            def ef(pos, vel, masses, G=1.0, eps=0.02):
                if pos.shape[0] < 5000:
                    try:
                        import src.nbody_numba as nb
                    except ImportError:
                        import nbody_numba as nb
                    return nb.total_energy_numba(pos, vel, masses, G, eps)
                return bh.total_energy_bh(pos, vel, masses, G, eps, theta=theta)
            return cf, ef, tag
        except Exception as e:
            print(f"[warn] Barnes-Hut indisponivel ({e}), fallback numba")
            try:
                import src.nbody_numba as nb
            except ImportError:
                import nbody_numba as nb
            return nb.compute_forces_numba, nb.total_energy_numba, "numba"
    if backend in ("numba", "parallel", "fast", "cpu"):
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        return nb.compute_forces_numba, nb.total_energy_numba, "numba"
    elif backend in ("gpu", "cupy", "cupy_fast", "cuda"):
        try:
            try:
                import src.nbody_gpu as gpu
            except ImportError:
                import nbody_gpu as gpu
            if gpu.cupy_available():
                # testa compilacao rapida para evitar crash de nvrtc no .venv sem toolkit
                try:
                    from src.nbody_gpu import compute_forces_cupy_fast as _cf
                except ImportError:
                    from nbody_gpu import compute_forces_cupy_fast as _cf
                # teste compilacao com N pequeno
                try:
                    _test_pos = np.zeros((4,3), dtype=np.float64)
                    _test_m = np.ones(4, dtype=np.float64)
                    _cf(_test_pos, _test_m)
                except Exception as ce:
                    raise ImportError(f"cupy compilacao falhou: {ce}")
                try:
                    import src.nbody_numba as nb
                except ImportError:
                    import nbody_numba as nb
                return _cf, nb.total_energy_numba, "cupy_fast"
            else:
                raise ImportError("cupy nao disponivel")
        except Exception as e:
            print(f"[warn] GPU cupy indisponivel ({e}), fallback numba")
            try:
                import src.nbody_numba as nb
            except ImportError:
                import nbody_numba as nb
            return nb.compute_forces_numba, nb.total_energy_numba, "numba"
    elif backend in ("gpu_persist", "cupy_persist", "gpu_fast_persist"):
        try:
            try:
                import src.nbody_gpu as gpu
            except ImportError:
                import nbody_gpu as gpu
            if gpu.cupy_available():
                try:
                    from src.nbody_gpu import compute_forces_cupy_fast_persistent as _cfp
                except ImportError:
                    from nbody_gpu import compute_forces_cupy_fast_persistent as _cfp
                try:
                    _test_pos = np.zeros((4,3), dtype=np.float64)
                    _test_m = np.ones(4, dtype=np.float64)
                    _cfp(_test_pos, _test_m)
                except Exception as ce:
                    raise ImportError(f"cupy persist compilacao falhou: {ce}")
                try:
                    import src.nbody_numba as nb
                except ImportError:
                    import nbody_numba as nb
                return _cfp, nb.total_energy_numba, "cupy_fast_persist"
            else:
                raise ImportError("cupy nao disponivel")
        except Exception as e:
            print(f"[warn] GPU cupy indisponivel ({e}), fallback numba")
            try:
                import src.nbody_numba as nb
            except ImportError:
                import nbody_numba as nb
            return nb.compute_forces_numba, nb.total_energy_numba, "numba"
    elif backend == "torch":
        try:
            try:
                from src.nbody_gpu import compute_forces_torch
            except ImportError:
                from nbody_gpu import compute_forces_torch
            try:
                import src.nbody_numba as nb
            except ImportError:
                import nbody_numba as nb
            return compute_forces_torch, nb.total_energy_numba, "torch"
        except Exception as e:
            print(f"[warn] torch indisponivel ({e}), fallback numba")
            try:
                import src.nbody_numba as nb
            except ImportError:
                import nbody_numba as nb
            return nb.compute_forces_numba, nb.total_energy_numba, "numba"
    elif backend in ("mpi", "mpi4py"):
        try:
            from src.nbody_mpi import compute_forces_mpi
        except ImportError:
            from nbody_mpi import compute_forces_mpi
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        return compute_forces_mpi, nb.total_energy_numba, "mpi"
    elif backend in ("mp", "multiprocessing", "mpi_mp"):
        try:
            from src.nbody_mpi import compute_forces_mp
        except ImportError:
            from nbody_mpi import compute_forces_mp
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        def cf_mp(pos, masses, G=1.0, eps=0.02):
            return compute_forces_mp(pos, masses, G, eps)
        return cf_mp, nb.total_energy_numba, "mp"
    elif backend == "seq":
        try:
            from src.nbody_sequential import compute_forces, total_energy
        except ImportError:
            from nbody_sequential import compute_forces, total_energy
        return compute_forces, total_energy, "seq"
    elif backend in ("ref", "loop", "reference"):
        try:
            from src.nbody_sequential import compute_forces_reference_loop, total_energy
        except ImportError:
            from nbody_sequential import compute_forces_reference_loop, total_energy
        return compute_forces_reference_loop, total_energy, "ref"
    elif backend == "tiled":
        try:
            from src.nbody_sequential import compute_forces_tiled as cf, total_energy_tiled as te
        except ImportError:
            from nbody_sequential import compute_forces_tiled as cf, total_energy_tiled as te
        return cf, te, "tiled"
    elif backend == "rebound":
        # REBOUND IAS15 — 15ª ordem adaptativo, estável para 400 luas com dt global pequeno
        # live_vispy com --backend rebound usa sim.integrate() direto, não compute_forces
        try:
            import src.nbody_numba as nb
        except ImportError:
            import nbody_numba as nb
        def cf_rebound(pos, masses, G=1.0, eps=0.01):
            return nb.compute_forces_numba(pos, masses, G, eps)
        return cf_rebound, nb.total_energy_numba, "rebound"
    else:
        raise ValueError(f"backend desconhecido: {backend}  use seq|numba|gpu|bh|mpi|mp|torch|rebound")

def run_fast(
    pos0: Vec3,
    vel0: Vec3,
    masses: npt.NDArray[np.float64],
    steps: int = 3000,
    dt: float = 0.005,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    backend: str = "numba",
    save_trajectory: bool = True,
    save_every: int = 2,
    energy_every: int = 50,
    verbose: bool = True,
    labels: Optional[npt.NDArray[np.int32]] = None,
) -> SimResult:
    compute_forces, total_energy, backend_tag = _get_backends(backend)
    pos = pos0.copy()
    vel = vel0.copy()
    N = pos.shape[0]
    cfg = SimConfig(N=N, steps=steps, dt=dt, G=G, eps=eps, integrator="leapfrog", tiled=False, seed=0)
    cfg.integrator = f"leapfrog-{backend}"  # type: ignore
    e0 = total_energy(pos, vel, masses, G, eps)
    energy_history = [e0]
    if verbose:
        print(f"[fast backend={backend} ({backend_tag}) N={N} steps={steps} dt={dt}] E0 kin={e0[0]:.4f} pot={e0[1]:.4f} tot={e0[2]:.4f}")
        if backend_tag == "numba":
            import numba
            print(f"  Numba threads: {numba.get_num_threads()}  ({numba.threading_layer()})")
        elif backend_tag == "cupy_fast":
            print(f"  GPU RTX 4060 Ti — cupy RawKernel (4352 CUDA cores)")
        elif backend_tag == "torch":
            print(f"  GPU torch CUDA")
    if save_trajectory:
        n_saved = steps // save_every + 1
        if steps % save_every != 0:
            n_saved += 1
        traj = np.zeros((n_saved, N, 3), dtype=np.float64)
        traj[0] = pos
        save_idx = 1
    else:
        traj = None
        save_idx = 0
    accel = compute_forces(pos, masses, G, eps)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        vel_half = vel + accel * (dt * 0.5)
        pos = pos + vel_half * dt
        accel_new = compute_forces(pos, masses, G, eps)
        vel = vel_half + accel_new * (dt * 0.5)
        accel = accel_new
        if save_trajectory and (step % save_every == 0):
            if save_idx < traj.shape[0]:
                traj[save_idx] = pos
                save_idx += 1
        if step % energy_every == 0 or step == steps:
            e = total_energy(pos, vel, masses, G, eps)
            energy_history.append(e)
            if verbose and (step % max(1, steps // 5) == 0 or step == steps):
                drift = abs(e[2] - e0[2]) / max(1e-12, abs(e0[2]))
                print(f"  step {step:5d}/{steps}  Etot={e[2]:.4f}  drift={drift:.2e}")
    if save_trajectory and save_idx < traj.shape[0]:
        traj[save_idx] = pos
        traj = traj[: save_idx + 1]
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  -> {elapsed:.2f}s  ({elapsed/steps*1000:.2f} ms/step)  backend={backend} ({backend_tag}) N={N}")
    result = SimResult(pos=pos, vel=vel, masses=masses, config=cfg, elapsed=elapsed, energy_history=energy_history, trajectory=traj)
    result.labels = labels  # type: ignore
    return result

def run_preset(preset: str = "galaxy", N: int = 1000, steps: int = 300, dt: float = 0.005, eps: float = 0.05, backend: str = "numba", seed: int = 42, verbose: bool = True, save_every: int = 2, n_asteroids: int = 0, asteroids_cache: str | None = None):
    """Helper para notebook: preset string -> (pos,vel,masses,labels) -> run_fast. Evita confusao de assinatura."""
    try:
        import src.presets as presets
    except ImportError:
        import presets as presets
    preset_names = None  # só solar_completo retorna nomes reais (JPL)
    if preset == "galaxy":
        pos, vel, masses, labels = presets.generate_spiral_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=seed)
    elif preset == "collision":
        pos, vel, masses, labels = presets.generate_galaxy_collision(seed=seed)
    elif preset == "plummer":
        pos, vel, masses = presets.generate_plummer_sphere(N, seed=seed); labels=None
    elif preset == "ring":
        pos, vel, masses, labels = presets.generate_ring_galaxy(N_disk=int(N*0.7), N_bulge=N-int(N*0.7), seed=seed)
    elif preset == "triple":
        pos, vel, masses, labels = presets.generate_triple_merger(N_per_galaxy=max(100, N//3), seed=seed)
    elif preset == "satellite":
        pos, vel, masses, labels = presets.generate_satellite_infall(N_host_disk=int(N*0.7), N_host_bulge=N-int(N*0.7)-200, N_sat=200, seed=seed)
    elif preset == "cold":
        pos, vel, masses = presets.generate_cold_collapse(N, seed=seed); labels=None
    elif preset == "kepler":
        pos, vel, masses, labels = presets.generate_kepler_binary(seed=seed)
    elif preset == "disk":
        pos, vel, masses, labels = presets.generate_kepler_disk(N, seed=seed)
    elif preset == "forming":
        pos, vel, masses, labels = presets.generate_forming_galaxy(N=N, seed=seed)
    elif preset == "solar":
        pos, vel, masses, labels = presets.generate_solar_system(seed=seed, with_moons=True)
    elif preset == "solar_real":
        pos, vel, masses, labels = presets.generate_solar_system_real(with_moons=True)
    elif preset == "solar_completo":
        pos, vel, masses, labels, names = presets.generate_solar_completo(with_moons=True, n_asteroids=n_asteroids, asteroids_cache=asteroids_cache)
        preset_names = names
    else:
        raise ValueError(preset)
    res = run_fast(pos, vel, masses, steps=steps, dt=dt, eps=eps, backend=backend, verbose=verbose, labels=labels, save_every=save_every)
    if preset_names is not None:
        res.names = preset_names  # nomes reais JPL, mesma ordem de masses/labels
    # compat: notebook espera (traj, masses, labels)
    return res.trajectory, res.masses, getattr(res, "labels", labels), res
