"""Testes de validação — rode com: .venv/Scripts/python -m pytest tests/ -v"""
import numpy as np
from src.nbody_sequential import (
    compute_forces, compute_forces_tiled, compute_forces_reference_loop,
    generate_initial_conditions, total_energy, total_energy_tiled,
    SimConfig, run_simulation,
)

def test_forces_vs_reference():
    pos, _, masses = generate_initial_conditions(16, seed=0)
    a_vec = compute_forces(pos, masses)
    a_ref = compute_forces_reference_loop(pos, masses)
    assert np.allclose(a_vec, a_ref, atol=1e-12)

def test_tiled_vs_vec():
    pos, _, masses = generate_initial_conditions(64, seed=1)
    for tile in [7, 16, 32, 64]:
        a_vec = compute_forces(pos, masses)
        a_tile = compute_forces_tiled(pos, masses, tile=tile)
        assert np.allclose(a_vec, a_tile, atol=1e-11), f"tile={tile}"

def test_energy_tiled():
    pos, vel, masses = generate_initial_conditions(128, seed=2)
    e1 = total_energy(pos, vel, masses)
    e2 = total_energy_tiled(pos, vel, masses)
    assert np.allclose(e1[2], e2[2], atol=1e-9)

def test_momentum_conservation():
    """Momento total deve permanecer ~0 (condição inicial + simetria das forças)."""
    pos, vel, masses = generate_initial_conditions(64, seed=3)
    # momento inicial ~0 por construção
    p0 = (masses[:, None] * vel).sum(axis=0)
    assert np.allclose(p0, 0, atol=1e-12)
    # após 10 steps leapfrog, momento deve conservar
    cfg = SimConfig(N=64, steps=10, dt=1e-4, integrator="leapfrog", seed=3)
    result = run_simulation(cfg, verbose=False)
    p1 = (result.masses[:, None] * result.vel).sum(axis=0)
    assert np.allclose(p1, 0, atol=1e-8), f"p1={p1}"

def test_energy_drift_leapfrog():
    """Leapfrog deve ter drift < 1% em 200 steps com dt=1e-4."""
    cfg = SimConfig(N=128, steps=200, dt=1e-4, integrator="leapfrog", seed=42)
    result = run_simulation(cfg, verbose=False, energy_every=50)
    e0 = result.energy_history[0][2]
    e1 = result.energy_history[-1][2]
    drift = abs(e1 - e0) / abs(e0)
    assert drift < 0.01, f"drift={drift:.2e} muito alto"

def test_two_body_circular():
    """Dois corpos em órbita circular aproximada devem manter distância ~constante."""
    G = 1.0; eps = 0.01
    pos = np.array([[0.5, 0, 0], [-0.5, 0, 0]], dtype=np.float64)
    # velocidade circular: v = sqrt(G*M / r) ; para M=1 cada, dist=1 => v ~ 1/sqrt(2) ??? simplificado com G=1, r=1, M=1 => v=1
    # Para 2 corpos iguais em órbita mútua: v = 0.5*sqrt(2) aprox
    vel = np.array([[0, 0.7, 0], [0, -0.7, 0]], dtype=np.float64)
    masses = np.array([1.0, 1.0])
    # usa leapfrog direto
    from src.nbody_sequential import compute_forces, step_leapfrog
    accel = compute_forces(pos, masses, G, eps)
    d0 = np.linalg.norm(pos[0]-pos[1])
    for _ in range(100):
        pos, vel, accel = step_leapfrog(pos, vel, accel, masses, dt=1e-4, G=G, eps=eps)
    d1 = np.linalg.norm(pos[0]-pos[1])
    assert abs(d1 - d0) / d0 < 0.05, f"d0={d0:.4f} d1={d1:.4f}"
