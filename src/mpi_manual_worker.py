"""
Worker MPI manual — servidor TCP persistente
Roda no CachyOS:  python -m src.mpi_manual_worker --port 5000 --threads 8
Master conecta 1× e reutiliza socket por todos os steps (evita SSH por frame).
"""
from __future__ import annotations
import argparse, socket, struct, pickle, time
import numpy as np

try:
    import numba
    @numba.njit(parallel=True)
    def _forces_slice(pos, masses, out, i0, i1, G, eps2):
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

def _send_pickle(sock, obj):
    data=pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!I", len(data)) + data)

def _recv_exact(sock, n):
    buf=b""
    while len(buf)<n:
        chunk=sock.recv(n-len(buf))
        if not chunk:
            return None
        buf+=chunk
    return buf

def _recv_pickle(sock):
    hdr=_recv_exact(sock, 4)
    if not hdr: return None
    n=struct.unpack("!I", hdr)[0]
    data=_recv_exact(sock, n)
    return pickle.loads(data) if data else None

def handle_client(conn, addr, threads):
    try:
        import numba
        numba.set_num_threads(threads)
    except: pass
    print(f"[worker] cliente {addr} threads={threads} (numba {threads})")
    if HAS_NUMBA:
        dummy_pos=np.zeros((4,3), dtype=np.float64)
        dummy_m=np.ones(4, dtype=np.float64)
        dummy_out=np.zeros((1,3), dtype=np.float64)
        _forces_slice(dummy_pos, dummy_m, dummy_out, 0, 1, 1.0, 0.0004)
    while True:
        try:
            job=_recv_pickle(conn)
            if job is None:
                print(f"[worker] {addr} desconectou")
                break
            pos=job["pos"]; masses=job["masses"]; i0=job["i0"]; i1=job["i1"]; G=job["G"]; eps=job["eps"]
            out=np.zeros((i1-i0,3), dtype=np.float64)
            if HAS_NUMBA:
                _forces_slice(pos, masses, out, i0, i1, G, eps*eps)
            else:
                try:
                    from src.nbody_sequential import compute_forces_tiled
                except ImportError:
                    from nbody_sequential import compute_forces_tiled
                full=compute_forces_tiled(pos, masses, G, eps)
                out[:]=full[i0:i1]
            _send_pickle(conn, out)
        except Exception as e:
            print(f"[worker] erro {addr}: {e}")
            break
    try: conn.close()
    except: pass

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--threads", type=int, default=8)
    args=p.parse_args()
    try:
        import numba
        numba.set_num_threads(args.threads)
        print(f"[worker] numba threads={numba.get_num_threads()}")
    except Exception as e:
        print(f"[worker] numba não configurado: {e}")
    srv=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except: pass
    srv.bind((args.host, args.port))
    srv.listen(8)
    print(f"[worker] ouvindo {args.host}:{args.port} threads={args.threads} (Ctrl+C para parar)")
    try:
        while True:
            conn, addr=srv.accept()
            try: conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except: pass
            conn.settimeout(None)
            import threading
            t=threading.Thread(target=handle_client, args=(conn, addr, args.threads), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[worker] encerrando")
    finally:
        srv.close()

if __name__=="__main__":
    main()
