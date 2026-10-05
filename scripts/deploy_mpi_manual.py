"""
Deploy worker MPI manual via SSH (1×, não por frame)
Uso:
  python scripts/deploy_mpi_manual.py --hosts hosts_manual.json
  # ou manual no worker:
  # CachyOS: python -m src.mpi_manual_worker --port 5000 --threads 8
"""
import json, pathlib, subprocess, sys, time

def deploy_one(host_cfg):
    host=host_cfg["host"]; port=host_cfg.get("port",5000)
    if host in ("localhost","127.0.0.1","::1"):
        print(f"[deploy] {host} local, pula SSH")
        return True
    user=host_cfg.get("user","user"); ssh_port=host_cfg.get("ssh_port", host_cfg.get("port",22))
    python=host_cfg.get("python","python")
    threads=host_cfg.get("threads",8)
    remote_cmd=f"nohup {python} -m src.mpi_manual_worker --port {port} --threads {threads} > /tmp/mpi_worker.log 2>&1 & echo $!"
    cmd=["ssh","-p",str(ssh_port), f"{user}@{host}", remote_cmd]
    print(f"[deploy] ssh {user}@{host} -p {ssh_port} -> worker :{port} ({threads}t)")
    try:
        proc=subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode==0:
            print(f"  ok pid {proc.stdout.strip()} log /tmp/mpi_worker.log")
            return True
        else:
            print(f"  fail {proc.stderr[:300]}")
            return False
    except Exception as e:
        print(f"  erro {e}")
        return False

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--hosts", default="hosts_manual.json")
    args=p.parse_args()
    hosts=json.loads(pathlib.Path(args.hosts).read_text(encoding="utf-8"))
    for h in hosts:
        deploy_one(h)
    print("Feito. Teste com: python -m src.nbody_mpi_manual --hosts hosts_manual.json --N 5000 --steps 2")
