"""
Visualização da simulação N-Body — VisPy por padrão, matplotlib fallback
=================================================================================
- plot_3d_snapshot / animate_trajectory: tenta VisPy GPU primeiro (60 FPS),
  cai para matplotlib se VisPy indisponível ou sem display (Colab headless).
  Para forçar matplotlib: `VISPY=0 python ...` ou `plot_3d_snapshot(..., use_vispy=False)`.
- plot_energy / plot_2d_projections: sempre matplotlib (2D).
- plot_3d_snapshot_vispy / animate_trajectory_vispy expostos para uso direto.

Live viewers dedicados: `scripts/live_vispy.py` (VisPy) e `scripts/live_pygame.py` (Pygame bloom).
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Configura ffmpeg via imageio-ffmpeg se disponível (resolve Colab/Windows sem ffmpeg no PATH)
try:
    import imageio_ffmpeg  # type: ignore
    import matplotlib

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

try:
    from src.nbody_sequential import SimResult
except ModuleNotFoundError:
    from nbody_sequential import SimResult

# VisPy opcional — não falha se ausente (Colab)
try:
    from src.visualization_vispy import _vispy_available, plot_3d_snapshot_vispy, animate_trajectory_vispy
except ImportError:
    try:
        from visualization_vispy import _vispy_available, plot_3d_snapshot_vispy, animate_trajectory_vispy
    except ImportError:
        _vispy_available = lambda: False
        plot_3d_snapshot_vispy = None  # type: ignore
        animate_trajectory_vispy = None  # type: ignore

import os as _os
_USE_VISPY_DEFAULT = _os.environ.get("VISPY", "1") != "0"


def plot_3d_snapshot(pos: np.ndarray, masses: np.ndarray | None = None, title: str = "N-Body — snapshot", save_path: str | None = None, labels: np.ndarray | None = None, colors: np.ndarray | None = None, use_vispy: bool | None = None):
    # VisPy por padrão local; Colab/headless cai para matplotlib automaticamente
    if use_vispy is None:
        use_vispy = _USE_VISPY_DEFAULT
    if use_vispy and _vispy_available():
        # VisPy precisa de display; se headless, fallback
        try:
            # tenta detectar headless (sem DISPLAY no Linux, ou dummy)
            if _os.environ.get("SDL_VIDEODRIVER") == "dummy":
                raise RuntimeError("headless")
            return plot_3d_snapshot_vispy(pos, masses, labels, title)
        except Exception:
            pass
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    # tamanho do ponto proporcional à massa
    if masses is not None:
        sizes = 5 + 20 * (masses - masses.min()) / max(1e-12, masses.max() - masses.min())
        # destaca BH se houver massa muito maior
        if masses.max() > 5 * np.median(masses):
            sizes = np.where(masses > 3 * np.median(masses), 80, sizes)
    else:
        sizes = 10
    if labels is not None:
        # colormap por labels (galáxia/colisão)
        uniq = np.unique(labels)
        cmap = plt.get_cmap("tab10", len(uniq))
        c = [cmap(int(np.where(uniq == l)[0][0])) for l in labels]
    elif colors is not None:
        c = colors
    else:
        c = np.linalg.norm(pos, axis=1)
        cmap = "viridis"
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, cmap=None if labels is not None or colors is not None else cmap, s=sizes, alpha=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    if labels is None and colors is None:
        fig.colorbar(sc, ax=ax, label="|r|")
    else:
        # legenda para labels
        if labels is not None:
            for ul in np.unique(labels):
                ax.scatter([], [], [], c=[plt.cm.tab10(ul % 10)], label=f"comp {ul}", s=30)
            ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Salvo: {save_path}")
    plt.show()
    return fig


def plot_2d_projections(pos: np.ndarray, labels: np.ndarray | None = None, title: str = "Projeções", save_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    projs = [("x", "y", (0, 1)), ("x", "z", (0, 2)), ("y", "z", (1, 2))]
    for ax, (lx, ly, (ix, iy)) in zip(axes, projs):
        if labels is not None:
            for ul in np.unique(labels):
                m = labels == ul
                ax.scatter(pos[m, ix], pos[m, iy], s=4, alpha=0.6, label=f"c{ul}")
        else:
            ax.scatter(pos[:, ix], pos[:, iy], s=4, c=np.linalg.norm(pos, axis=1), cmap="viridis", alpha=0.6)
        ax.set_xlabel(lx); ax.set_ylabel(ly); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
    axes[0].legend(markerscale=2)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Salvo: {save_path}")
    plt.show()
    return fig


def plot_energy(result: SimResult, save_path: str | None = None, preset: str | None = None):
    hist = np.array(result.energy_history)  # (T,3)
    steps = np.linspace(0, result.config.steps, len(hist))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Energia ao longo do tempo
    axes[0].plot(steps, hist[:, 0], label="E_kin")
    axes[0].plot(steps, hist[:, 1], label="E_pot")
    axes[0].plot(steps, hist[:, 2], label="E_tot", linewidth=2)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("Energia")
    preset_tag = f" {preset}" if preset else ""
    axes[0].set_title(f"Energia —{preset_tag} N={result.config.N} {result.config.integrator}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Drift relativo
    e0 = hist[0, 2]
    drift = np.abs(hist[:, 2] - e0) / max(1e-12, abs(e0))
    axes[1].semilogy(steps, drift + 1e-16)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("|E(t)-E0|/|E0|")
    axes[1].set_title(f"Drift de energia (log) —{preset_tag} {result.config.integrator}")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Salvo: {save_path}")
    plt.show()
    return fig


def plot_energy_comparison(results: dict[str, SimResult], save_path: str | None = None, preset: str | None = None):
    """
    Compara drift de energia entre integradores/presets.
    results: dict label -> SimResult (ex: {"leapfrog": res_lf, "euler": res_eu})
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for label, res in results.items():
        hist = np.array(res.energy_history)
        steps = np.linspace(0, res.config.steps, len(hist))
        e0 = hist[0, 2]
        drift = np.abs(hist[:, 2] - e0) / max(1e-12, abs(e0))
        axes[0].semilogy(steps, drift + 1e-16, label=f"{label} ({res.config.integrator})", linewidth=2)
        axes[1].plot(steps, hist[:, 2], label=f"E_tot {label}")
    preset_tag = f" — {preset}" if preset else ""
    axes[0].set_xlabel("step"); axes[0].set_ylabel("|E(t)-E0|/|E0|")
    axes[0].set_title(f"Drift energia — Euler vs Leapfrog{preset_tag} (log)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("E_total")
    axes[1].set_title(f"E_total —{preset_tag}")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Salvo: {save_path}")
    plt.show()
    return fig


def animate_trajectory(result: SimResult, save_path: str | None = None, interval: int = 40, fps: int = 20, labels: np.ndarray | None = None, trails: bool = False, elev: int = 20, azim: int = 30, use_vispy: bool | None = None, glow: bool = False, size: float = 6.0):
    """
    Requer result.trajectory != None (run_simulation(save_trajectory=True)).
    - labels: array (N,) para colorir por galáxia/componente (tab10)
    - trails: desenha rastro dos últimos 10 frames (opcional, mais pesado) — só matplotlib
    - use_vispy: None=auto (VisPy se disponível), True força VisPy, False força matplotlib
    - glow/size: só VisPy (bloom whole-scene, mass->size)
    """
    if use_vispy is None:
        use_vispy = _USE_VISPY_DEFAULT and not trails  # trails ainda só matplotlib
    if use_vispy and _vispy_available() and animate_trajectory_vispy is not None:
        try:
            if _os.environ.get("SDL_VIDEODRIVER") == "dummy":
                raise RuntimeError("headless")
            return animate_trajectory_vispy(result.trajectory, result.masses, labels if labels is not None else getattr(result, "labels", None), dt=getattr(result.config, "dt", 0.005), title=f"N={result.config.N} {result.config.integrator}", save_path=save_path, size=size, glow=glow)
        except Exception:
            pass
    if result.trajectory is None:
        print("Sem trajectory — rode run_simulation(save_trajectory=True)")
        return None

    traj = result.trajectory  # (S, N, 3)  S = saved steps
    T, N, _ = traj.shape
    # tenta pegar labels do result se não passado
    if labels is None and hasattr(result, "labels") and getattr(result, "labels") is not None:
        labels = getattr(result, "labels")
    # limites fixos para câmera não pular
    all_pos = traj.reshape(-1, 3)
    # percentil para evitar outlier explodir limites
    lim = np.percentile(np.abs(all_pos), 99) * 1.4
    lim = max(lim, 1.0)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    masses = result.masses
    # tamanho destaca BH
    median_m = np.median(masses)
    sizes = 5 + 12 * (masses - masses.min()) / max(1e-12, masses.max() - masses.min())
    if masses.max() > 4 * median_m:
        sizes = np.where(masses > 3 * median_m, 60, sizes)

    # cores por labels ou por partícula se N pequeno
    if labels is not None:
        uniq = np.unique(labels)
        cmap = plt.get_cmap("tab10", len(uniq))
        # mapeia label->cor
        label_to_color = {int(u): cmap(i) for i, u in enumerate(uniq)}
        point_colors = np.array([label_to_color[int(l)] for l in labels])
    else:
        point_colors = plt.cm.tab20(np.linspace(0, 1, N)) if N <= 30 else None
        if point_colors is None:
            # cor por raio inicial para visual interessante
            point_colors = np.linalg.norm(traj[0], axis=1)

    p0 = traj[0]
    # para color array fixo, passa como c
    if labels is not None:
        sc = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=sizes, alpha=0.9, c=point_colors)
    else:
        if point_colors is not None and point_colors.ndim == 2:
            sc = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=sizes, alpha=0.85, c=point_colors)
        else:
            sc = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=sizes, alpha=0.6, c=point_colors, cmap="viridis" if point_colors is not None else None)

    # rastro opcional (linhas)
    trail_lines = []
    if trails and labels is not None:
        for ul in np.unique(labels):
            (line,) = ax.plot([], [], [], alpha=0.25, linewidth=0.8)
            trail_lines.append(line)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    title = ax.set_title("")
    # legenda
    if labels is not None:
        for ul in np.unique(labels):
            ax.scatter([], [], [], c=[label_to_color[int(ul)]], label=f"comp {int(ul)}", s=30)
        ax.legend(loc="upper right", fontsize=8)

    def init():
        sc._offsets3d = ([], [], [])
        return (sc,)

    def update(frame):
        p = traj[frame]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        # atualiza trails
        if trails and labels is not None:
            # mostra últimos 15 frames
            start = max(0, frame - 15)
            for line, ul in zip(trail_lines, np.unique(labels)):
                m = labels == ul
                # pega trajetória do centro de massa do componente?
                # simplesmente desenha linhas de todas partículas do componente seria pesado
                # desenha centro de massa
                com_traj = traj[start:frame+1, m, :].mean(axis=1)  # (window,3)
                line.set_data(com_traj[:, 0], com_traj[:, 1])
                line.set_3d_properties(com_traj[:, 2])
        title.set_text(f"N={N} frame {frame}/{T-1}  t={frame*getattr(result.config,'dt',0):.3f}")
        return (sc,)

    # stride para não gerar vídeo muito longo
    stride = max(1, T // 400)
    frames = range(0, T, stride)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval, blit=False)

    if save_path:
        path = pathlib.Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # para N grande ou muitos frames, reduz dpi para não travar
        save_dpi = 100 if (N > 1000 or T > 400) else 150
        try:
            ani.save(str(path), writer="ffmpeg", fps=fps, dpi=save_dpi)
            print(f"Vídeo salvo: {path} (dpi={save_dpi})")
        except KeyboardInterrupt:
            print("Vídeo interrompido pelo usuário (Ctrl+C) — simulação já salva em data/*.png")
            plt.close(fig)
            return ani
        except Exception as e:
            gif_path = path.with_suffix(".gif")
            print(f"ffmpeg falhou ({e}), salvando gif: {gif_path}")
            try:
                ani.save(str(gif_path), writer="pillow", fps=fps, dpi=100)
                print(f"Gif salvo: {gif_path}")
            except KeyboardInterrupt:
                print("Gif interrompido — use --no_video para só ver interativo (muito mais rápido)")
                plt.close(fig)
                return ani

    plt.show()
    return ani


def plot_performance_comparison(results: dict, save_path: str | None = None):
    """
    results: dict N -> elapsed  ou dict label -> (N, elapsed)
    Para uso nas Fases 2-5.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, data in results.items():
        # data pode ser lista de (N, elapsed) ou dict
        if isinstance(data, dict):
            Ns = sorted(data.keys())
            times = [data[n] for n in Ns]
        else:
            Ns, times = zip(*data)
        ax.loglog(Ns, times, marker="o", label=label)
        # linha de referência O(N²)
        ref = [times[0] * (n / Ns[0]) ** 2 for n in Ns]
        ax.loglog(Ns, ref, linestyle="--", alpha=0.4)
    ax.set_xlabel("N"); ax.set_ylabel("tempo (s)")
    ax.set_title("Escalabilidade N-Body — referência O(N²) tracejada")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig
