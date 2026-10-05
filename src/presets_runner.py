"""
Runner genérico para presets — integra com nbody_sequential leapfrog.
Permite salvar trajetória e energia para qualquer (pos,vel,masses) custom.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    from src.nbody_sequential import (
        compute_forces, compute_forces_tiled, total_energy, total_energy_tiled,
        step_leapfrog, step_euler, SimConfig, SimResult, G_DEFAULT, EPS_DEFAULT
    )
except ImportError:
    from nbody_sequential import (
        compute_forces, compute_forces_tiled, total_energy, total_energy_tiled,
        step_leapfrog, step_euler, SimConfig, SimResult, G_DEFAULT, EPS_DEFAULT
    )

Vec3 = npt.NDArray[np.float64]


def run_custom_simulation(
    pos0: Vec3,
    vel0: Vec3,
    masses: npt.NDArray[np.float64],
    steps: int = 1000,
    dt: float = 1e-3,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    integrator: str = "leapfrog",
    tiled: bool = False,
    save_trajectory: bool = True,
    save_every: int = 1,
    energy_every: int = 10,
    verbose: bool = True,
    labels: Optional[npt.NDArray[np.int32]] = None,
) -> SimResult:
    """
    Similar a run_simulation mas parte de estado custom.
    save_every: salva trajetória a cada K steps (para economizar RAM).
    labels: opcional, repassado via config-like para visualização.
    """
    pos = pos0.copy()
    vel = vel0.copy()
    N = pos.shape[0]

    # Config fake para SimResult
    cfg = SimConfig(N=N, steps=steps, dt=dt, G=G, eps=eps, integrator=integrator, tiled=tiled, seed=0)

    energy_history = []
    # energia inicial
    if N <= 2000 and not tiled:
        e0 = total_energy(pos, vel, masses, G, eps)
    else:
        e0 = total_energy_tiled(pos, vel, masses, G, eps)
    energy_history.append(e0)
    if verbose:
        print(f"[custom N={N} steps={steps} dt={dt} {integrator}{' tiled' if tiled else ''}] E0 kin={e0[0]:.4f} pot={e0[1]:.4f} tot={e0[2]:.4f}")

    # trajetória: aloca com stride
    if save_trajectory:
        n_saved = steps // save_every + 1
        traj = np.zeros((n_saved, N, 3), dtype=np.float64)
        traj[0] = pos
        save_idx = 1
    else:
        traj = None
        save_idx = 0

    if integrator == "leapfrog":
        accel = compute_forces_tiled(pos, masses, G, eps) if (tiled or N > 2000) else compute_forces(pos, masses, G, eps)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if integrator == "euler":
            # euler precisa recalcular a cada step
            try:
                from src.nbody_sequential import step_euler
            except ImportError:
                from nbody_sequential import step_euler
            pos, vel = step_euler(pos, vel, masses, dt, G, eps, tiled=tiled or N > 2000)
        else:
            pos, vel, accel = step_leapfrog(pos, vel, accel, masses, dt, G, eps, tiled=tiled or N > 2000)

        if save_trajectory and (step % save_every == 0):
            traj[save_idx] = pos
            save_idx += 1
        # se save_every não divide steps, garante último frame
        if save_trajectory and step == steps and (step % save_every != 0):
            traj[-1] = pos

        if step % energy_every == 0 or step == steps:
            if N <= 2000 and not tiled:
                e = total_energy(pos, vel, masses, G, eps)
            else:
                e = total_energy_tiled(pos, vel, masses, G, eps)
            energy_history.append(e)
            if verbose and (step % max(1, steps // 10) == 0 or step == steps):
                drift = abs(e[2] - e0[2]) / max(1e-12, abs(e0[2]))
                print(f"  step {step:5d}/{steps}  Etot={e[2]:.4f}  drift={drift:.2e}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  -> {elapsed:.2f}s  ({elapsed/steps*1000:.2f} ms/step) N={N}")

    # guarda labels dentro de masses? Não, cria atributo extra via SimResult estendido
    result = SimResult(pos=pos, vel=vel, masses=masses, config=cfg, elapsed=elapsed, energy_history=energy_history, trajectory=traj)
    # anexa labels para visualização se fornecido
    if labels is not None:
        result.labels = labels  # type: ignore
        # também salva labels originais da trajetória? Usamos mesmo labels para todos os tempos
    else:
        result.labels = None  # type: ignore
    return result
