r"""
6 animações 3D VisPy em sequência — sem cap de 17 FPS do matplotlib
Cada uma usa mesmo preset 3D interativo (arraste rotaciona) mas backend diferente.
Feche a janela para ir para a próxima e ver fluidez aumentar.

Uso:
  .venv312\Scripts\python scripts/archive/live_vispy_6.py                 # N=1500 collision
  .venv312\Scripts\python scripts/archive/live_vispy_6.py --N 3000 --steps 3000
"""
import pathlib, sys, subprocess, time
ROOT = pathlib.Path(__file__).resolve().parents[2]

# 6 cenários: N² seq (pior) -> N² par CPU -> N² par GPU -> BH par CPU th0.9 -> BH th0.5 -> BH th0.7 (ou GPU BH se existisse)
# Para 3D VisPy todos ficam 60 FPS visual, mas compute FPS mostra diferença real (0.3ms vs 200ms)
cmds = [
    ("1/6  N² seq CPU (pior, ~200ms/step, VisPy ainda 60 FPS mas física lenta — corpos andam devagar)",
     [sys.executable, "scripts/live_vispy.py", "--preset", "collision", "--backend", "seq", "--N", "800", "--steps", "1500"]),
    ("2/6  N² par CPU (numba 28t, ~1ms, fluido)",
     [sys.executable, "scripts/live_vispy.py", "--preset", "collision", "--backend", "numba", "--N", "1500", "--steps", "2000"]),
    ("3/6  N² par GPU (cupy RTX, ~0.3ms, máximo)",
     [sys.executable, "scripts/live_vispy.py", "--preset", "collision", "--backend", "gpu", "--N", "1500", "--steps", "2000"]),
    ("4/6  BH par CPU th=0.9 (O(N log N) rápido, ~30ms, 30 FPS mas permite N=5000)",
     [sys.executable, "scripts/live_vispy.py", "--preset", "plummer", "--backend", "bh", "--N", "3000", "--steps", "1500"]),
    ("5/6  BH par CPU th=0.7 (preciso, ~40ms)",
     [sys.executable, "scripts/live_vispy.py", "--preset", "plummer", "--backend", "bh0.7", "--N", "3000", "--steps", "1500"]),
    ("6/6  BH par CPU th=0.5 (mais preciso, ~70ms)",
     [sys.executable, "scripts/live_vispy.py", "--preset", "plummer", "--backend", "bh0.5", "--N", "3000", "--steps", "1500"]),
]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=None)
parser.add_argument("--steps", type=int, default=None)
args = parser.parse_args()

for title, cmd in cmds:
    # ajusta N/steps se passado via CLI
    if args.N is not None:
        # substitui --N valor
        for i, v in enumerate(cmd):
            if v == "800" or v == "1500" or v == "3000":
                # heurística: mantém proporção
                pass
        # simplifica: reescreve --N
        if "--N" in cmd:
            idx = cmd.index("--N")
            cmd[idx+1] = str(args.N)
    if args.steps is not None and "--steps" in cmd:
        idx = cmd.index("--steps")
        cmd[idx+1] = str(args.steps)
    print("\n"+"="*70)
    print(title)
    print(" ".join(cmd))
    print("Feche a janela 3D (VisPy) para ir para a próxima — FPS no título é real (compute+render VisPy ~2ms)")
    subprocess.run(cmd, cwd=ROOT)
    time.sleep(1)

print("\nFim das 6 — VisPy 3D sem cap de 17 FPS do matplotlib (OpenGL GPU).")
print("Para 2D máximo 144 FPS use live_pygame.py, para matplotlib clássico use live_demo.py")
