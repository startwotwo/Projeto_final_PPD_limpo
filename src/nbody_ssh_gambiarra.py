"""
Pseudo-MPI via SSH — Gambiarra pedagógica heterogênea
======================================================
Master (Windows i7 14k 28t) + Worker (CachyOS i7 930 8t) sem mpi4py.
Divide N por i (row-wise) + Allgatherv via SSH pickle.

Uso:
  python -m src.nbody_ssh_gambiarra --N 5000 --steps 100 --hosts hosts.json
  hosts.json: [{"host":"localhost","threads":28},{"host":"200.18.98.132","port":2222,"user":"user","threads":8,"python":".venv/bin/python"}]

Comunicação: SSH exec "python -" no worker, envia (pos,masses,i0,i1,G,eps) pickle via stdin, recebe accel[i0:i1] pickle via stdout.
Latência ~5ms vs MPI 0.02ms — ótimo para relatório de granularidade.
"""
from __future__ import annotations
import json, pathlib, subprocess, pickle, sys, time, os
import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]

# Compilado 1x no topo — reutilizado por todos os steps/hosts (evita 1s recompile por Allgatherv)
try:
    import numba
    @numba.njit
    def _forces_slice_numba(pos, masses, out, i0, i1, G, eps2):
        N = pos.shape[0]
        for i in range(i0, i1):
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
    HAS_NUMBA_SLICE=True
except Exception:
    HAS_NUMBA_SLICE=False

WORKER_CODE = r'''
import sys, pickle, numpy as np
import numba
@numba.njit(parallel=True)
def _forces_slice(pos, masses, out, i0, i1, G, eps2):
    N = pos.shape[0]
    for i in range(i0, i1):
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
data = pickle.load(sys.stdin.buffer)
pos=data['pos']; masses=data['masses']; i0=data['i0']; i1=data['i1']; G=data['G']; eps=data['eps']
out=np.zeros((i1-i0,3), dtype=np.float64)
_forces_slice(pos, masses, out, i0, i1, G, eps*eps)
sys.stdout.buffer.write(pickle.dumps(out))
'''

def _run_local_slice(pos, masses, i0, i1, G, eps):
    if HAS_NUMBA_SLICE:
        out=np.zeros((i1-i0,3), dtype=np.float64)
        _forces_slice_numba(pos, masses, out, i0, i1, G, eps*eps)
        return out
    from .nbody_sequential import compute_forces_tiled
    full=compute_forces_tiled(pos, masses, G, eps)
    return full[i0:i1]

_worker_deployed=set()
def _ensure_worker_simple(host_cfg):
    host=host_cfg["host"]; port=host_cfg.get("port",22); user=host_cfg.get("user","user")
    if host in _worker_deployed or host in ("localhost","127.0.0.1"):
        return
    cmd=["ssh","-p",str(port), f"{user}@{host}", "cat > /tmp/gambiarra_worker.py && chmod +x /tmp/gambiarra_worker.py && echo ok"]
    proc=subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err=proc.communicate(input=WORKER_CODE.encode(), timeout=10)
    if proc.returncode!=0:
        raise RuntimeError(f"Deploy {host} fail: {err.decode()[:500]}")
    _worker_deployed.add(host)

VERBOSE_SSH=False
def _run_ssh_slice(host_cfg, pos, masses, i0, i1, G, eps):
    host=host_cfg["host"]; port=host_cfg.get("port",22); user=host_cfg.get("user","user")
    py=host_cfg.get("python","python")
    if VERBOSE_SSH:
        import datetime as _dt
        print(f"[{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]  host {host} ({host_cfg.get('threads')}t) começou i0={i0} n={i1-i0}", flush=True)
        t_host=time.perf_counter()
    if host in ("localhost","127.0.0.1"):
        res=_run_local_slice(pos, masses, i0, i1, G, eps)
    else:
        _ensure_worker_simple(host_cfg)
        cmd=["ssh","-p",str(port), f"{user}@{host}", f"{py} /tmp/gambiarra_worker.py"]
        data={"pos":pos, "masses":masses, "i0":int(i0), "i1":int(i1), "G":float(G), "eps":float(eps)}
        pickled=pickle.dumps(data, protocol=4)
        proc=subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err=proc.communicate(input=pickled, timeout=30)
        if proc.returncode!=0:
            raise RuntimeError(f"SSH {host} fail: {err.decode()[:500]}")
        res=pickle.loads(out)
    if VERBOSE_SSH:
        import datetime as _dt
        print(f"[{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]  host {host} terminou em {(time.perf_counter()-t_host)*1000:.1f} ms", flush=True)
    return res

def compute_forces_ssh_gambiarra(pos, masses, G=1.0, eps=0.02, hosts=None):
    if hosts is None:
        p=pathlib.Path("hosts.json")
        if p.exists():
            hosts=json.loads(p.read_text(encoding="utf-8"))
        else:
            hosts=[{"host":"localhost","threads":28}]
    total_threads=sum(h.get("threads",1) for h in hosts)
    N=pos.shape[0]
    counts=[]
    for h in hosts:
        w=h.get("threads",1)/total_threads
        counts.append(int(round(N*w)))
    diff=N-sum(counts)
    counts[0]+=diff
    slices=[]
    start=0
    for c in counts:
        end=start+c
        slices.append((start,end))
        start=end
    import concurrent.futures
    results=[None]*len(hosts)
    def _task(idx):
        h=hosts[idx]; i0,i1=slices[idx]
        if i0>=i1:
            return idx, np.zeros((0,3))
        return idx, _run_ssh_slice(h, pos, masses, i0, i1, G, eps)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        futs=[ex.submit(_task,i) for i in range(len(hosts))]
        for f in concurrent.futures.as_completed(futs):
            idx,res=f.result()
            results[idx]=res
    return np.vstack(results)

if __name__=="__main__":
    import argparse
    from .presets import generate_spiral_galaxy
    p=argparse.ArgumentParser(description="Pseudo-MPI SSH gambiarra")
    p.add_argument("--N", type=int, default=2000)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--hosts", type=str, default="hosts.json")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.005)
    args=p.parse_args()
    hosts=None
    if pathlib.Path(args.hosts).exists():
        hosts=json.loads(pathlib.Path(args.hosts).read_text(encoding="utf-8"))
        print(f"Hosts: {hosts}")
    else:
        print(f"hosts.json não encontrado, usando localhost 28t")
        hosts=[{"host":"localhost","threads":28}]
    pos, vel, masses, _= generate_spiral_galaxy(N_disk=int(args.N*0.7), N_bulge=args.N-int(args.N*0.7), seed=42)
    from .nbody_sequential import compute_forces_tiled
    import time
    t0=time.perf_counter()
    a_ref=compute_forces_tiled(pos, masses, eps=args.eps)
    t_ref=(time.perf_counter()-t0)*1000
    print(f"[{time.strftime('%H:%M:%S')}] N={args.N} tiled {t_ref:.1f} ms (1 host 28t)")
    print(f"[{time.strftime('%H:%M:%S')}] Testando ssh_gambiarra com {len(hosts)} hosts...")
    VERBOSE_SSH=True
    t0=time.perf_counter()
    a_ssh=compute_forces_ssh_gambiarra(pos, masses, eps=args.eps, hosts=hosts)
    t_ssh=(time.perf_counter()-t0)*1000
    err=np.max(np.abs(a_ref-a_ssh))
    print(f"[{time.strftime('%H:%M:%S')}] N={args.N} ssh_gambiarra {t_ssh:.1f} ms max err {err:.2e} {'PASS' if err<1e-9 else 'FAIL'}")
    t0=time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Começou simulação {args.steps} steps")
    accel=compute_forces_ssh_gambiarra(pos, masses, eps=args.eps, hosts=hosts)
    for step in range(args.steps):
        print(f"[{time.strftime('%H:%M:%S')}] Começou step {step+1}/{args.steps}")
        vel_half=vel+accel*args.dt*0.5
        pos=pos+vel_half*args.dt
        t_step=time.perf_counter()
        accel=compute_forces_ssh_gambiarra(pos, masses, eps=args.eps, hosts=hosts)
        vel=vel_half+accel*args.dt*0.5
        import datetime as _dt2
        print(f"[{_dt2.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]  step {step+1} Allgatherv em {(time.perf_counter()-t_step)*1000:.1f} ms (master esperou straggler)")
        if step%max(1,args.steps//5)==0:
            print(f"[{time.strftime('%H:%M:%S')}] step {step}/{args.steps} completo")
    print(f"Simulação {args.steps} steps em {time.perf_counter()-t0:.1f}s N={args.N} hosts={len(hosts)}")
