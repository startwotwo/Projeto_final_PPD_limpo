"""
N-Body backend paralelo — Threads CPU (Numba prange/OpenMP)
============================================================
Compila loops em C via @njit(parallel=True) + prange para 28 threads.
Mesma física que nbody_sequential.py, mas com paralelismo real nos cores.
"""
# blz vamo lá. como funciona o numba:
# ela é uma bin de paralelização pro pyhton que usa OpenMP por baixo
# não existe openMP pro python, então pra paralelizar com thread e tal, o interessante é usar esse
# é meio bunda porque a gente meio que não faz nada, o numba faz sozinho toda essa parte de cálculo e paralelização
# MAAAAS não sou eu quem faz as regras né 
# a mariana que pediu pra usar python 🥱

# a grande questão é a seguinte: 

from __future__ import annotations

import os
import numpy as np
import numpy.typing as npt

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

Vec3 = npt.NDArray[np.float64]

if HAS_NUMBA:
    @numba.njit(parallel=True, fastmath=True)
    def compute_forces_numba(
        pos: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        G: float,
        eps: float,
    ) -> npt.NDArray[np.float64]:
        N = pos.shape[0]
        """ (porra eu acabei de descobrir que dá pra usar essas três aspas duplas juntas pra fazer comentário de bloco em qualquer lugar)
        tem que lembrar que tudo na vida tem um revés né
        nesse caso fatídico é que apesar do numpy vetorizar, ele tem a limitação do formato das matrizes

        é muito fodido esse numba
        mas ele é nuito foda (pq ele usa o LLVM)
        ele e o clang 
        são dois muito fodas que que não sabia que eram tão foda
        como funciona: o nosso grande objetivo aqui são 2: 
        - paralelizar em várias threads/processos
        - VETORIZAR operações quando possível (e vamos fazer isso com SIMD - single instruction multiple data)
        por isso, a gente vai usar o numba, porque ele faz os dois:
        - ele transforma o código do python (python bytecode) quando a gente põe o @njit (njit = no-python just-in-time) em código executável diretamente na 
        memória usando o LLVM (Low Level Virtual Machine):
            - o numba primeiro faz uma limpeza das bostas que o python deixa (tipo não definido, listas, etc) e converte pra Numba IR (Intermediate Representation)
            - o LLVM pega o código em numba IR (na verdade é uma camada de compatibildade python-llvm, chama llvmlite, que é o que o numba usa, pq o LLVM de verdade é gigante,
            daí ele carrega só essa casquinha ao invés de carregar o LLVM inteiro toda vez)
            - transforma de código numba IR pra LLVM IR, que é uma linguagem de baixo nível nível que é UNIVERSAL pras arquiteturas de CPU
            - com o LLVM dá pra você pegar qualquer código, pode ser python, rust, C/C++, e transformar pra IR. Cada um tem o seu (Clang pro C, rustc pro rust, etc)
            - o LLVM (que é escrito em C++) e transforma esse código IR e começa a fazer a otimização pesadona (loop unrolling, vetorização pra SIMD, inlining de 
            função, etc. Parece com o -O3 do gcc)
            - depois da super otimização, o LLVM IR otimizado vira instrução de máquina (não é mais assembly, isso já é passado) binário mesmo, e é executado direto na RAM
            (o LLVM nesse momento também verifica qual é a plataforma que ele tá fazendo o código (x86-64, ARM, etc) e mapeia os dados pros registradores, etc)

        além disso, tem outra coisa:
        rigorosamente, aquele "compute_forces_reference_loop", apesar de ser só demonstrativo, também pode ser otimizado fazendo uns ajustes, que é o que eu vou fazer aqui
        """
        accel = np.zeros((N, 3), dtype=np.float64)
        eps2 = eps * eps
        for i in numba.prange(N):
            # ao invés de alocar vetores inteiros, alocar só as posições
            ax = 0.0
            ay = 0.0
            az = 0.0
            xi = pos[i, 0]  # pega só a pos x de i
            yi = pos[i, 1]  # pega só a pos y de i
            zi = pos[i, 2]  # pega só a pos z de i
            for j in range(N):
                if i == j:
                    continue

                # faz todos os cálculos com escalares ao invés de vetores
                # quando eu faço assim eu carrego pontos específicos da memória (floats)
                # ao invés de ficar criando novos objetos vetor python (r_vec = pos[j] - pos[i])
                # é uma otimização que é simples e poderia ser implementada no caso lá do sequencial
                # mas como o objetivo é avaliar só esses piores casos mais zoados, acho que é melhor assim
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                dz = pos[j, 2] - zi
                r2 = dx * dx + dy * dy + dz * dz + eps2
                inv_r = 1.0 / np.sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r
                s = G * masses[j] * inv_r3
                ax += s * dx
                ay += s * dy
                az += s * dz
            accel[i, 0] = ax
            accel[i, 1] = ay
            accel[i, 2] = az
        return accel

    @numba.njit(parallel=True, fastmath=True)
    def total_energy_numba(
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        G: float,
        eps: float,
    ) -> tuple[float, float, float]:
        N = pos.shape[0]
        ekin = 0.0
        for i in numba.prange(N):
            # módulo das velocidades ao quadrado, igualzinho o do numpy, só que pegando só os 
            # pontos específicos
            # de novo, isso só dá pra fazer assim porque o numba vai vetorizar isso tudo pra mim 
            v2 = vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1] + vel[i, 2] * vel[i, 2]
            ekin += 0.5 * masses[i] * v2
        epot = 0.0
        eps2 = eps * eps
        for i in numba.prange(N):
            for j in range(i + 1, N):
                # módulo da distância 
                # aqui era aquele monte de matriz e operação com matriz pra fazer no numpy
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                r = np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
                epot -= G * masses[i] * masses[j] / r
        return ekin, epot, ekin + epot

    @numba.njit 
    def _leapfrog_step_numba_inner(
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        accel: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # vou implementar só o leapfrog, porque a esse ponto já tá mais que explicado porque o do 
        # euler é muito mais bosta que ele
        # ele não precisa paralelizar em thread porque só tem uma conta por vez
        vel_half = vel + accel * (dt * 0.5)
        pos_new = pos + vel_half * dt
        return pos_new, vel_half

else:

    def compute_forces_numba(*args, **kwargs):
        raise RuntimeError("Numba não instalado")

    def total_energy_numba(*args, **kwargs):
        raise RuntimeError("Numba não instalado")


def compute_forces(pos: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02) -> Vec3:
    if not HAS_NUMBA:
        try:
            from .nbody_sequential import compute_forces as seq
        except ImportError:
            from nbody_sequential import compute_forces as seq
        return seq(pos, masses, G, eps)
    return compute_forces_numba(pos, masses, G, eps)


def total_energy(pos: Vec3, vel: Vec3, masses: npt.NDArray[np.float64], G: float = 1.0, eps: float = 0.02):
    if not HAS_NUMBA:
        try:
            from .nbody_sequential import total_energy as seq
        except ImportError:
            from nbody_sequential import total_energy as seq
        return seq(pos, vel, masses, G, eps)
    return total_energy_numba(pos, vel, masses, G, eps)


def info() -> dict:
    if not HAS_NUMBA:
        return {"has_numba": False}
    return {
        "has_numba": True,
        "numba_version": numba.__version__,
        "threads": numba.get_num_threads(),
        "threading_layer": numba.threading_layer(),
        "cuda_available": False,
    }


if __name__ == "__main__":
    print(info())
    import time
    from .presets import generate_spiral_galaxy

    pos, vel, masses, _ = generate_spiral_galaxy(N_disk=400, N_bulge=200, seed=42)
    compute_forces_numba(pos, masses, 1.0, 0.05)
    t0 = time.perf_counter()
    for _ in range(20):
        compute_forces_numba(pos, masses, 1.0, 0.05)
    elapsed = (time.perf_counter() - t0) / 20 * 1000
    print(f"compute_forces_numba N={len(masses)}  {elapsed:.2f} ms/call  threads={numba.get_num_threads()}")
