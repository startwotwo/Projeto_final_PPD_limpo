"""
VisPy visualization — N-Body 3D (padrão local, Colab fallback matplotlib)
============================================================================
Usa GPU via VisPy (Markers + turntable) — 60 FPS para N=5000 vs matplotlib 2 FPS.
API compatível com visualization.py, mas sem dependência de mpl_toolkits/mpl 3D.
"""
from __future__ import annotations
import numpy as np

def _vispy_available() -> bool:
    try:
        import vispy  # noqa
        from vispy import scene, app  # noqa
        return True
    except Exception:
        return False

def _mass_to_sizes(masses, base=6.0):
    if masses is None:
        return np.full(100, base, dtype=np.float32)  # dummy
    m_med = np.median(masses)
    sizes = (base * (masses / max(m_med, 1e-12)) ** 0.33).astype(np.float32)
    return np.clip(sizes, base*0.6, base*3.0)

def plot_3d_snapshot_vispy(pos, masses=None, labels=None, title="N-Body VisPy", bgcolor="black"):
    """Snapshot VisPy interativo — arraste para rotacionar, scroll zoom."""
    if not _vispy_available():
        # fallback
        from .visualization import plot_3d_snapshot as mpl_snap
        return mpl_snap(pos, masses, title, labels=labels)
    from vispy import scene, app
    import numpy as np
    N = pos.shape[0]
    # cores por labels
    if labels is not None:
        uniq = np.unique(labels)
        tab10 = np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189]], dtype=float)/255
        colors = np.array([tab10[int(np.where(uniq==l)[0][0])%5] for l in labels], dtype=np.float32)
        colors = np.column_stack([colors, np.full(N, 0.95)])
    else:
        colors = np.ones((N,4), dtype=np.float32)
        colors[:,:3] = 0.8
    sizes = _mass_to_sizes(masses, base=6.0)
    canvas = scene.SceneCanvas(keys='interactive', show=True, size=(900,700), bgcolor=bgcolor, title=title)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = max(8.0, np.percentile(np.abs(pos), 99)*1.4)
    view.camera.center = (0,0,0)
    scatter = scene.visuals.Markers(scaling='fixed', antialias=0)
    scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
    view.add(scatter)
    # HUD
    text = scene.visuals.Text(title, color="white", font_size=9, pos=(10,20), parent=canvas.scene)
    print(f"VisPy snapshot N={N} — arraste para rotacionar, feche para continuar")
    app.run()
    return canvas

def animate_trajectory_vispy(trajectory, masses=None, labels=None, dt=0.005, title="N-Body", save_path=None, size=6.0, glow=False, speed=1.0):
    """
    Animação VisPy live — trajectory shape (T,N,3). Usa lógica de live_vispy.py
    com bloom opcional e mass->size.
    """
    if not _vispy_available():
        # fallback matplotlib
        from .visualization import animate_trajectory as mpl_anim
        # precisa SimResult wrapper — cria dummy
        from .nbody_sequential import SimConfig, SimResult
        cfg = SimConfig(N=trajectory.shape[1], steps=trajectory.shape[0]-1, dt=dt)
        res = SimResult(pos=trajectory[-1], vel=np.zeros_like(trajectory[-1]), masses=masses if masses is not None else np.ones(trajectory.shape[1]), config=cfg, elapsed=0, energy_history=[], trajectory=trajectory)
        if labels is not None:
            res.labels = labels
        return mpl_anim(res, save_path=save_path)
    from vispy import scene, app
    import time
    from PIL import Image, ImageFilter
    T, N, _ = trajectory.shape
    if masses is None:
        masses = np.ones(N, dtype=np.float64)
    # cores
    if labels is not None:
        uniq = np.unique(labels)
        tab10 = np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189]], dtype=float)/255
        colors = np.array([tab10[int(np.where(uniq==l)[0][0])%5] for l in labels], dtype=np.float32)
        colors = np.column_stack([colors, np.full(N, 0.95)])
    else:
        colors = np.ones((N,4), dtype=np.float32)
    m_med = np.median(masses)
    sizes = (size * (masses / max(m_med,1e-12))**0.33).astype(np.float32)
    sizes = np.clip(sizes, size*0.6, size*3.0)
    canvas = scene.SceneCanvas(keys='interactive', show=True, size=(900,700), bgcolor="black", title=title)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 12
    lim = np.percentile(np.abs(trajectory.reshape(-1,3)), 99)*1.3
    lim = max(lim, 3.0)
    view.camera.center = (0,0,0)
    scatter = scene.visuals.Markers(scaling='fixed', antialias=0)
    scatter.set_data(trajectory[0].astype(np.float32), face_color=colors, size=sizes, edge_width=0)
    view.add(scatter)
    scatter.order = 0
    # bloom behind
    bloom_img = scene.visuals.Image(parent=canvas.scene, method='subdivide')
    bloom_img.set_gl_state('translucent', blend=True, blend_func=('src_alpha','one'))
    bloom_img.visible = False
    step = 0
    t_last = time.perf_counter(); fps_avg=60
    def update(ev):
        nonlocal step, t_last, fps_avg
        if step >= T:
            if save_path:
                print(f"Anim fim T={T}")
            app.quit()
            return
        pos = trajectory[step]
        scatter.set_data(pos.astype(np.float32), face_color=colors, size=sizes, edge_width=0)
        if glow and step % 3 == 0:
            try:
                img = canvas.render(alpha=True)
                pil = Image.fromarray(img)
                W2, H2 = pil.size
                small = pil.resize((W2//2, H2//2), Image.BILINEAR)
                small = small.filter(ImageFilter.GaussianBlur(radius=2))
                bloom = small.resize((W2, H2), Image.BILINEAR)
                bloom_np = (np.array(bloom).astype(np.float32)*0.6).astype(np.uint8)
                bloom_img.set_data(bloom_np)
                bloom_img.visible = True
            except Exception:
                pass
        elif not glow:
            bloom_img.visible = False
        canvas.update()
        dt_total = time.perf_counter() - t_last
        t_last = time.perf_counter()
        fps_avg = 0.9*fps_avg + 0.1*(1/max(dt_total,1e-6))
        canvas.title = f"{title}  frame {step}/{T}  FPS {fps_avg:.0f}  {'bloom' if glow else ''}"
        step += max(1, int(speed))
    timer = app.Timer(interval=0.0, connect=update, start=True)
    print(f"VisPy animate T={T} N={N} glow={glow} speed={speed} — feche para continuar")
    app.run()
    return canvas
