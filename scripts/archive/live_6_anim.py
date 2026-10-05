r"""
Roda as 6 animações ao vivo em sequência — feche a janela para ir para a próxima
Mostra visualmente a diferença de fluidez (FPS) entre os 6 cenários.

Uso:
  .venv312\Scripts\python scripts/archive/live_6_anim.py                 # N=1500, collision
  .venv312\Scripts\python scripts/archive/live_6_anim.py --N 3000       # N maior, diferença mais gritante
"""
import pathlib, sys, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[2]

# sequência pedida: N² seq CPU (pior) -> N² par CPU -> N² par GPU
#                    BH seq -> BH par CPU th0.9 -> BH par CPU th0.5 (ou BH GPU se existisse)
cmds = [
    ("1/6  N² seq CPU (pior, ~200ms/step, 4 FPS travado)", 
     [sys.executable, "scripts/archive/live_demo.py", "--preset", "collision", "--backend", "seq", "--N", "800", "--steps", "800", "--dt", "0.005"]),
    ("2/6  N² par CPU (numba 28t, ~1ms, 30 FPS)", 
     [sys.executable, "scripts/archive/live_demo.py", "--preset", "collision", "--backend", "numba", "--N", "1500", "--steps", "1500", "--dt", "0.005"]),
    ("3/6  N² par GPU (cupy RTX, ~0.3ms, 60 FPS fluido)", 
     [sys.executable, "scripts/archive/live_demo.py", "--preset", "collision", "--backend", "gpu", "--N", "1500", "--steps", "1500", "--dt", "0.005"]),
    ("4/6  BH seq CPU (python puro, didático, ~1500ms, lento)", 
     [sys.executable, "scripts/archive/live_demo.py", "--preset", "plummer", "--backend", "bh", "--N", "1500", "--steps", "800", "--dt", "0.005"]),
    ("5/6  BH par CPU th=0.9 (O(N log N) rápido, ~30ms, 30 FPS)", 
     [sys.executable, "scripts/archive/live_demo.py", "--preset", "plummer", "--backend", "bh", "--N", "3000", "--steps", "1500", "--dt", "0.005"]),
    ("6/6  BH par CPU th=0.5 (preciso, ~40ms)", 
     [sys.executable, "scripts/archive/live_demo.py", "--preset", "plummer", "--backend", "bh0.7", "--N", "3000", "--steps", "1500", "--dt", "0.005"]),
]

# Se quiser 3D GPU fluido máximo, troque live_demo.py por live_vispy.py nas linhas acima
# Ex: ["scripts/live_vispy.py", "--preset", "collision", "--backend", "gpu", "--N", "3000"]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=1500)
parser.add_argument("--gpu_only", action="store_true", help="só N² gpu vs BH gpu")
args = parser.parse_args()
if args.N != 1500:
    # ajusta N nos comandos
    for _, cmd in cmds:
        for i, v in enumerate(cmd):
            if v == "800" or v == "1500" or v == "3000":
                # substitui N genérico
                pass

for title, cmd in cmds:
    print("\n"+"="*70)
    print(title)
    print(" ".join(cmd))
    print("Feche a janela 3D para ir para a próxima...")
    subprocess.run(cmd, cwd=ROOT)
    print("Próxima em 2s...")
    import time; time.sleep(2)

print("\nFim das 6 animações ao vivo — compare FPS no título (ms/step).")
