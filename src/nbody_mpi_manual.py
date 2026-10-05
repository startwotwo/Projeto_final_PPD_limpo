"""
MPI manual sem MS-MPI/OpenMPI — sockets TCP persistentes
==========================================================
Substitui nbody_ssh_gambiarra.py (SSH por frame → 5 ms) por
conexões TCP mantidas abertas (latência ~0.05 ms, 100× melhor).

Arquitetura:
  Master (Windows)                     Worker (Linux/CachyOS)
  - hosts.json: [{"host":"200.18.98.132","port":5000,"threads":8}]
  - No início: master faz socket.create_connection((host,port)) 1× por worker,
    mantém em pool. Worker roda `python -m src.mpi_manual_worker --port 5000`.
  - Por step: master envia (pos,masses,i0,i1,G,eps) via pickle+header 4B,
    worker calcula _forces_slice numba e devolve (i1-i0,3) pickle.
    ThreadPoolExecutor faz Allgatherv em paralelo.
  - Sem SSH, sem mpirun, só Python stdlib (socket, pickle, struct).

Decomposição 1-D row-wise + Allgather, idêntica a nbody_mpi.py:96
Cada rank calcula accel[i0:i1] = Σ_j G m_j r_ij/(r²+eps²)^{3/2}

Uso:
  # Worker (CachyOS):
  python -m src.mpi_manual_worker --port 5000 --threads 8
  # Master (Windows):
  python -m src.nbody_mpi_manual --N 5000 --steps 100 --hosts hosts.json
  hosts.json: [{"host":"localhost","threads":28},{"host":"200.18.98.132","port":5000,"threads":8}]

Fallback: se hosts.json ausente ou só localhost → usa multiprocessing.Pool local
sem sockets (mesma API).
"""
from __future__ import annotations
import json, pathlib, socket, struct, pickle, time, os, threading
import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor

Vec3 = npt.NDArray[np.float64]

# ------------------------------------------------------------------
# Núcleo numba reutilizado (igual nbody_mpi.py:37)
# ------------------------------------------------------------------
try:
    import numba
    @numba.njit(parallel=True)
    def _forces_slice_numba(pos, masses, out, i0, i1, G, eps2):
        N = pos.shape[0]
        for i in numba.prange(i0, i1):
            ax=ay=az=0.0
            xi, yi, zi = pos[i,0], pos[i,1], pos[i,2]
            for j in range(N):
                if i==j: continue
                dx=pos[j,0]-xi; dy=pos[j,1]-yi; dz=pos[j,2]-zi
                r2=dx*dx+dy*dy+dz*dz+eps2
                inv_r=1.0/np.sqrt(r2); inv_r3=inv_r*inv_r*inv_r
                s=G*masses[j]*inv_r3
                ax+=s*dx; ay+=s*dy; az+=s*dz
            out[i-i0,0]=ax; out[i-i0,1]=ay; out[i-i0,2]=az
    HAS_NUMBA=True
except Exception:
    HAS_NUMBA=False
    def _forces_slice_numba(pos, masses, out, i0, i1, G, eps2):
        raise RuntimeError("numba ausente")

def _forces_slice(pos_all, masses, i0, i1, G, eps):
    if HAS_NUMBA:
        out=np.zeros((i1-i0,3), dtype=np.float64)
        _forces_slice_numba(pos_all, masses, out, i0, i1, G, eps*eps)
        return out
    try:
        from .nbody_sequential import compute_forces_tiled
    except ImportError:
        from nbody_sequential import compute_forces_tiled
    full=compute_forces_tiled(pos_all, masses, G, eps)
    return full[i0:i1]

# ------------------------------------------------------------------
# Protocolo pickle + header 4B (evita TCP fragmentation)
# ------------------------------------------------------------------
def _send_pickle(sock: socket.socket, obj):
    data=pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!I", len(data)) + data)

def _recv_pickle(sock: socket.socket):
    hdr=_recv_exact(sock, 4)
    if not hdr:
        return None
    n=struct.unpack("!I", hdr)[0]
    data=_recv_exact(sock, n)
    return pickle.loads(data) if data else None

def _recv_exact(sock, n):
    buf=b""
    while len(buf)<n:
        chunk=sock.recv(n-len(buf))
        if not chunk:
            return None
        buf+=chunk
    return buf

# ------------------------------------------------------------------
# Pool de conexões persistentes
# ------------------------------------------------------------------
_conn_pool: dict[str, socket.socket] = {}
_pool_lock = threading.Lock()

def _get_conn(host_cfg):
    key=f"{host_cfg['host']}:{host_cfg.get('port',5000)}"
    with _pool_lock:
        if key in _conn_pool:
            try:
                _conn_pool[key].getpeername()
                _conn_pool[key].settimeout(15.0)
                return _conn_pool[key]
            except Exception:
                try: _conn_pool[key].close()
                except: pass
                del _conn_pool[key]
        host=host_cfg["host"]; port=int(host_cfg.get("port",5000))
        sock=socket.create_connection((host, port), timeout=5.0)
        sock.settimeout(15.0)
        try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except: pass
        _conn_pool[key]=sock
        return sock

def _close_all():
    with _pool_lock:
        for s in _conn_pool.values():
            try: s.close()
            except: pass
        _conn_pool.clear()

import atexit; atexit.register(_close_all)

VERBOSE_MPI=False
# ------------------------------------------------------------------
# Execução por fatia (local ou remota)
# ------------------------------------------------------------------
def _run_slice(host_cfg, pos, masses, i0, i1, G, eps):
    if VERBOSE_MPI:
        import datetime as _dt
        print(f"[{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]  host {host_cfg['host']} ({host_cfg.get('threads')}t) começou i0={i0} n={i1-i0}", flush=True)
        t_host=time.perf_counter()
    if host_cfg["host"] in ("localhost","127.0.0.1","::1"):
        res=_forces_slice(pos, masses, i0, i1, G, eps)
    else:
        sock=_get_conn(host_cfg)
        try:
            _send_pickle(sock, {"pos":pos, "masses":masses, "i0":int(i0), "i1":int(i1), "G":float(G), "eps":float(eps)})
            resp=_recv_pickle(sock)
            if resp is None:
                raise RuntimeError(f"Worker {host_cfg['host']} fechou conexão")
            res=resp
        except Exception as e:
            key=f"{host_cfg['host']}:{host_cfg.get('port',5000)}"
            with _pool_lock:
                if key in _conn_pool:
                    try: _conn_pool[key].close()
                    except: pass
                    _conn_pool.pop(key, None)
            sock=_get_conn(host_cfg)
            _send_pickle(sock, {"pos":pos, "masses":masses, "i0":int(i0), "i1":int(i1), "G":float(G), "eps":float(eps)})
            resp=_recv_pickle(sock)
            if resp is None:
                raise RuntimeError(f"Retry falhou {host_cfg['host']}: {e}")
            res=resp
    if VERBOSE_MPI:
        import datetime as _dt
        print(f"[{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]  host {host_cfg['host']} terminou em {(time.perf_counter()-t_host)*1000:.1f} ms", flush=True)
    return res

def _run_local_slice(pos, masses, i0, i1, G, eps):
    return _forces_slice(pos, masses, i0, i1, G, eps)

# ------------------------------------------------------------------
# API principal — Allgatherv manual
# ------------------------------------------------------------------
def compute_forces_mpi_manual(pos, masses, G=1.0, eps=0.02, hosts=None):
    if hosts is None:
        p=pathlib.Path("hosts.json")
        if p.exists():
            hosts=json.loads(p.read_text(encoding="utf-8"))
        else:
            hosts=[{"host":"localhost","threads":28}]
    total_threads=sum(int(h.get("threads",1)) for h in hosts)
    N=pos.shape[0]
    counts=[]
    for h in hosts:
        w=int(h.get("threads",1))/total_threads
        counts.append(int(round(N*w)))
    diff=N-sum(counts)
    if diff!=0:
        idx_max=max(range(len(hosts)), key=lambda i: hosts[i].get("threads",1))
        counts[idx_max]+=diff
    slices=[]
    start=0
    for c in counts:
        end=start+max(0,c)
        slices.append((start,end))
        start=end
    results=[None]*len(hosts)
    def task(idx):
        h=hosts[idx]; i0,i1=slices[idx]
        if i0>=i1:
            return idx, np.zeros((0,3), dtype=np.float64)
        return idx, _run_slice(h, pos, masses, i0, i1, G, eps)
    with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        futs=[ex.submit(task,i) for i in range(len(hosts))]
        for f in futs:
            idx,res=f.result()
            results[idx]=res
    vs=[r for r in results if r is not None and r.size>0]
    if not vs:
        return np.zeros((N,3), dtype=np.float64)
    return np.vstack(vs)

def compute_forces_mp_manual(pos, masses, G=1.0, eps=0.02, nproc=None):
    import multiprocessing as mp, os
    nproc=nproc or os.cpu_count() or 4
    N=pos.shape[0]
    counts=[N//nproc + (1 if r < N%nproc else 0) for r in range(nproc)]
    displs=[sum(counts[:r]) for r in range(nproc)]
    args=[(pos,masses,displs[r],displs[r]+counts[r],G,eps) for r in range(nproc)]
    with mp.Pool(nproc) as pool:
        parts=pool.starmap(_forces_slice, args)
    return np.vstack(parts)

def run_simulation_mpi_manual(N=2000, steps=200, dt=0.005, eps=0.05, G=1.0, seed=42, preset="galaxy", hosts=None, verbose=True):
    import time
    global VERBOSE_MPI
    try:
        from .presets import generate_spiral_galaxy, generate_galaxy_collision, generate_plummer_sphere
    except ImportError:
        from presets import generate_spiral_galaxy, generate_galaxy_collision, generate_plummer_sphere
    if hosts is None:
        p=pathlib.Path("hosts.json")
        hosts=json.loads(p.read_text(encoding="utf-8")) if p.exists() else [{"host":"localhost","threads":28}]
    if preset=="galaxy":
        pos,vel,masses,_=generate_spiral_galaxy(N_disk=int(N*0.7),N_bulge=N-int(N*0.7),seed=seed)
    elif preset=="collision":
        pos,vel,masses,_=generate_galaxy_collision(seed=seed); N=len(masses)
    elif preset=="plummer":
        pos,vel,masses=generate_plummer_sphere(N,seed=seed)
    else:
        pos,vel,masses,_=generate_spiral_galaxy(N_disk=int(N*0.7),N_bulge=N-int(N*0.7),seed=seed)
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] N={N} hosts={hosts}")
        VERBOSE_MPI=True
    accel=compute_forces_mpi_manual(pos,masses,G,eps,hosts=hosts)
    t0=time.perf_counter()
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Começou simulação {steps} steps")
    for step in range(steps):
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Começou step {step+1}/{steps}")
        vel_half=vel+accel*(dt*0.5)
        pos=pos+vel_half*dt
        t_step=time.perf_counter()
        accel=compute_forces_mpi_manual(pos,masses,G,eps,hosts=hosts)
        vel=vel_half+accel*(dt*0.5)
        if verbose:
            import datetime as _dt2
            print(f"[{_dt2.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]  step {step+1} Allgatherv em {(time.perf_counter()-t_step)*1000:.1f} ms (master esperou straggler)")
            if step%max(1,steps//5)==0:
                print(f"[{time.strftime('%H:%M:%S')}] step {step+1}/{steps} completo")
    elapsed=time.perf_counter()-t0
    if verbose:
        print(f"MPI-manual N={N} steps={steps} {elapsed:.2f}s {elapsed/steps*1000:.1f} ms/step hosts={len(hosts)}")
    VERBOSE_MPI=False
    return pos,vel,masses,elapsed

if __name__=="__main__":
    import argparse, time
    p=argparse.ArgumentParser(description="MPI manual socket persistente")
    p.add_argument("--N",type=int,default=2000); p.add_argument("--steps",type=int,default=50)
    p.add_argument("--hosts",type=str,default="hosts.json"); p.add_argument("--eps",type=float,default=0.05)
    p.add_argument("--dt",type=float,default=0.005)
    args=p.parse_args()
    hosts=json.loads(pathlib.Path(args.hosts).read_text(encoding="utf-8")) if pathlib.Path(args.hosts).exists() else [{"host":"localhost","threads":28}]
    print(f"Hosts: {hosts}")
    try:
        from src.nbody_sequential import compute_forces_tiled
    except ImportError:
        from nbody_sequential import compute_forces_tiled
    try:
        from .presets import generate_spiral_galaxy
    except ImportError:
        from presets import generate_spiral_galaxy
    pos,vel,masses,_=generate_spiral_galaxy(N_disk=int(args.N*0.7),N_bulge=args.N-int(args.N*0.7),seed=42)
    t0=time.perf_counter(); a_ref=compute_forces_tiled(pos,masses,eps=args.eps); t_ref=(time.perf_counter()-t0)*1000
    print(f"[{time.strftime('%H:%M:%S')}] N={args.N} tiled {t_ref:.1f} ms (1 host 28t)")
    print(f"[{time.strftime('%H:%M:%S')}] Testando mpi-manual com {len(hosts)} hosts...")
    VERBOSE_MPI=True
    t0=time.perf_counter(); a_man=compute_forces_mpi_manual(pos,masses,eps=args.eps,hosts=hosts); t_man=(time.perf_counter()-t0)*1000
    err=float(np.max(np.abs(a_ref-a_man)))
    print(f"[{time.strftime('%H:%M:%S')}] N={args.N} mpi-manual {t_man:.1f} ms err {err:.2e} {'PASS' if err<1e-9 else 'FAIL'}")
    VERBOSE_MPI=False
    run_simulation_mpi_manual(N=args.N, steps=args.steps, eps=args.eps, dt=args.dt, hosts=hosts)
