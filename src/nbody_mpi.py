"""
N-Body MPI — Memória distribuída

Decomposição 1-D por partículas (row-wise) + Allgather.

Estratégia adequada ao O(N²) direto:
 - Cada rank p detém fatia i ∈ [p*N/P , (p+1)*N/P) e calcula
   accel[i] = sum_j G m_j r_ij/(r²+eps²)^{3/2}
 - Para calcular forças precisa de TODAS as posições a cada step
   => Allgather de pos (N,3) no início do step (custo O(N log P)).
   Comunicação domina para N pequeno; para N grande computação O(N²/P) domina.
 - Balanceamento trivial (N/P igual, mesmo custo por i). Granularidade
   grossa: 1 Allgather por step, mensagem única de 24*N bytes.
 - Alternativa hierárquica (Barnes-Hut + ORB) seria melhor para N>50k,
   mas foge ao escopo da disciplina


"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]

def _has_mpi() -> bool:
    try:
        import mpi4py  # noqa
        return True
    except Exception:
        return False

# ------------------------------------------------------------------
# Núcleo local: fatia de i (reusa numba se disponível)
# ------------------------------------------------------------------
def _forces_slice(pos_all: Vec3, masses: npt.NDArray[np.float64], i0: int, i1: int, G: float, eps: float) -> Vec3:
    try:
        import numba
        @numba.njit
        def _inner(pos, masses, out, i0, i1, G, eps2):
            N = pos.shape[0]
            for i in range(i0, i1):
                ax=ay=az=0.0
                xi, yi, zi = pos[i,0], pos[i,1], pos[i,2]
                for j in range(N):
                    if i==j: continue
                    dx = pos[j,0]-xi; dy = pos[j,1]-yi; dz = pos[j,2]-zi
                    r2 = dx*dx+dy*dy+dz*dz+eps2
                    inv_r = 1.0/np.sqrt(r2); inv_r3 = inv_r*inv_r*inv_r
                    s = G*masses[j]*inv_r3
                    ax+=s*dx; ay+=s*dy; az+=s*dz
                out[i-i0,0]=ax; out[i-i0,1]=ay; out[i-i0,2]=az
        out=np.zeros((i1-i0,3),dtype=np.float64)
        _inner(pos_all,masses,out,i0,i1,G,eps*eps)
        return out
    except Exception:
        # fallback numpy tiled para fatia
        try:
            from .nbody_sequential import compute_forces_tiled
        except ImportError:
            from nbody_sequential import compute_forces_tiled
        # calcula tudo e fatia (menos eficiente mas correto)
        full = compute_forces_tiled(pos_all,masses,G,eps,tile=1024)
        return full[i0:i1]


# API MPI real (mpi4py)
# infelizmente isso só funciona se a gente tem o MPI por fora. O mpi4py é só um wrapper pro mpi que tem que ter instalado na máquina
# eu até ia instalar essa porra pra rodar nos dois PC do lab, mas eu descobri que os SO precisam ser o mesmo 
# pro windows é o MS-MPI e pro linux é o OpenMPI ou o MPICH, então de todo jeito não daria certo

# mas, por questões de curiosidade da minha parte, eu vou fazer uma puta gambiarra com SSH mesmo pra pelo menos conseguir usar 2 pcs diferentes
# "a mas o outro pc é uma bosta" fodase
# nas palavras do Emerson: "VAI FICAR BOM"
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⡻⠻⡟⢛⡛⣿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⠑⣢⠶⢃⢭⣀⣤⣀⡈⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣃⢈⠜⢅⣴⣾⣿⣿⣿⣿⣿⣿⡆⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣪⣡⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠢⢺⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⡿⣿⢛⣉⣉⡙⠻⣿⡏⣉⣉⣀⡙⡜⢾⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣏⢱⣿⣞⣉⣸⣦⣿⣿⣿⣽⣥⣃⣣⣵⢘⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⢼⣿⣿⣿⣿⣿⣿⣿⣿⣻⣿⣿⣿⣿⢰⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣵⣧⣶⣽⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⡛⠋⣉⡉⠉⣩⣾⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⡿⢻⣿⠿⣿⣿⣿⣿⣿⠟⠋⡀⠘⢿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⠿⠛⠀⠀⠻⣿⣶⣦⣤⣤⣤⣰⡿⠁⠀⠀⠉⠛⠻⢿⣿⣿⣿⣿
# ⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀⠀⠻⣿⣟⣿⣿⣿⣏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠛
# ⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢩⠙⢿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀




def compute_forces_mpi(pos: Vec3, masses: npt.NDArray[np.float64], G: float=1.0, eps: float=0.02) -> Vec3:
    """
    Deve ser chamada dentro de mpirun; usa Allgather implícito porque
    pos já é replicado em todos os ranks antes da chamada.
    Cada rank calcula apenas sua fatia e faz Allgatherv do resultado.
    Se chamada sem MPI (size==1) equivale ao sequencial.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        # sem mpi4py -> sequencial
        try:
            from .nbody_sequential import compute_forces_tiled
        except ImportError:
            from nbody_sequential import compute_forces_tiled
        return compute_forces_tiled(pos,masses,G,eps)
    N = pos.shape[0]
    # fatia 1-D balanceada
    counts = [N//size + (1 if r < N%size else 0) for r in range(size)]
    displs = [sum(counts[:r]) for r in range(size)]
    i0, i1 = displs[rank], displs[rank]+counts[rank]
    local = _forces_slice(pos,masses,i0,i1,G,eps)  # (cnt,3)
    # Allgatherv
    recv = np.empty((N,3),dtype=np.float64)
    # mpi4py precisa de counts em elementos (cnt*3)
    scounts = [c*3 for c in counts]; sdispls=[d*3 for d in displs]
    comm.Allgatherv(local.ravel(), [recv.ravel(), scounts, sdispls, MPI.DOUBLE])
    return recv


# Fallback multiprocessing (emula passagem de mensagem sem mpirun)

def compute_forces_mp(pos: Vec3, masses: npt.NDArray[np.float64], G: float=1.0, eps: float=0.02, nproc: int|None=None) -> Vec3:
    import multiprocessing as mp
    nproc = nproc or os.cpu_count() or 4
    N = pos.shape[0]
    counts = [N//nproc + (1 if r < N%nproc else 0) for r in range(nproc)]
    displs = [sum(counts[:r]) for r in range(nproc)]
    args = [(pos,masses,displs[r],displs[r]+counts[r],G,eps) for r in range(nproc)]
    # pos é broadcast (pickle) => modela Allgather (custo O(N*P))
    with mp.Pool(nproc) as pool:
        parts = pool.starmap(_forces_slice, args)
    return np.vstack(parts)

# Simulação completa MPI (leapfrog simplético)
def run_simulation_mpi(N: int=2000, steps: int=200, dt: float=0.005, eps: float=0.05, G: float=1.0, seed: int=42, preset: str="galaxy", verbose: bool=True):
    try:
        from .presets import generate_spiral_galaxy, generate_galaxy_collision, generate_plummer_sphere, generate_cold_collapse
    except ImportError:
        from presets import generate_spiral_galaxy, generate_galaxy_collision, generate_plummer_sphere, generate_cold_collapse
    try:
        from mpi4py import MPI
        comm=MPI.COMM_WORLD; rank=comm.Get_rank()
    except Exception:
        rank=0; comm=None
    # rank 0 gera IC e broadcast
    if rank==0:
        if preset=="galaxy":
            pos,vel,masses,_=generate_spiral_galaxy(N_disk=int(N*0.7),N_bulge=N-int(N*0.7),seed=seed)
        elif preset=="collision":
            pos,vel,masses,_=generate_galaxy_collision(seed=seed)
            N=len(masses)  # ignora N arg
        elif preset=="plummer":
            pos,vel,masses=generate_plummer_sphere(N,seed=seed)
        else:
            pos,vel,masses,_=generate_spiral_galaxy(N_disk=int(N*0.7),N_bulge=N-int(N*0.7),seed=seed)
    else:
        pos=vel=masses=None
    if comm is not None:
        pos=comm.bcast(pos,root=0); vel=comm.bcast(vel,root=0); masses=comm.bcast(masses,root=0); N=comm.bcast(N,root=0)
    accel=compute_forces_mpi(pos,masses,G,eps)
    t0=time.perf_counter()
    for step in range(steps):
        # leapfrog: precisa Allgather de pos a cada step
        vel_half=vel+accel*(dt*0.5)
        pos=pos+vel_half*dt
        if comm is not None:
            # broadcast do novo pos (emula Allgather implícito: cada rank driftou igual)
            # Na verdade todos calculam mesma atualização, então basta manter replicado
            pass
        accel=compute_forces_mpi(pos,masses,G,eps)
        vel=vel_half+accel*(dt*0.5)
        if verbose and rank==0 and step%max(1,steps//5)==0:
            print(f" step {step}/{steps}")
    elapsed=time.perf_counter()-t0
    if rank==0 and verbose:
        print(f"MPI run N={N} steps={steps}  {elapsed:.2f}s  {elapsed/steps*1000:.1f} ms/step")
    return pos,vel,masses,elapsed

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--N",type=int,default=2000); p.add_argument("--steps",type=int,default=100)
    p.add_argument("--preset",default="galaxy"); p.add_argument("--mp",type=int,default=0,help="usa Pool mp em vez de mpi4py (colab)")
    p.add_argument("--eps",type=float,default=0.05)
    a=p.parse_args()
    if a.mp>0:
        try:
            from .presets import generate_spiral_galaxy
        except ImportError:
            from presets import generate_spiral_galaxy
        pos,vel,masses,_=generate_spiral_galaxy(N_disk=int(a.N*0.7),N_bulge=a.N-int(a.N*0.7),seed=42)
        t0=time.perf_counter()
        for _ in range(5):
            compute_forces_mp(pos,masses,eps=a.eps,nproc=a.mp)
        print(f"MP {a.mp}  {(time.perf_counter()-t0)/5*1000:.1f} ms/call N={a.N}")
    else:
        run_simulation_mpi(N=a.N,steps=a.steps,preset=a.preset)
