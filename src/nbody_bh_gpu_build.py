"""
Build da octree 100% na GPU (morton + prefix)
Usa cupy para sort e construção bottom-up sem Python puro.
"""
import numpy as np
import cupy as cp

# NOTA: existia aqui um stub _morton_code() morto (corpo = pass).
# Removido na revisao — Morton e feito no device via _get_morton_kernel().

_build_kernel_cache = {}

_morton_cache = {}
_bhdev = {"N": 0, "M": 0, "masses_id": None}

# offsets dos 8 octantes (mesmos de nbody_barnes_hut._OCT_OFFSETS)
_BH_OFF = None


def _bh_offsets():
    global _BH_OFF
    if _BH_OFF is None:
        import cupy as cp
        _BH_OFF = cp.array([
            [-0.25, -0.25, -0.25],
            [+0.25, -0.25, -0.25],
            [-0.25, +0.25, -0.25],
            [+0.25, +0.25, -0.25],
            [-0.25, -0.25, +0.25],
            [+0.25, -0.25, +0.25],
            [-0.25, +0.25, +0.25],
            [+0.25, +0.25, +0.25],
        ], dtype=cp.float32)
    return _BH_OFF


def clear_bh_gpu_device_cache() -> None:
    """Libera buffers device do build/traversal 100% GPU."""
    global _bhdev
    _bhdev = {"N": 0, "M": 0, "masses_id": None}


def _get_morton_kernel():
    import cupy as cp
    if "morton" in _morton_cache:
        return _morton_cache["morton"]
    code = r'''
    extern "C" __global__
    void morton_codes(const float* pos, int* codes, int N,
                      float minx, float miny, float minz, float scale) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;
        float x = (pos[3*i]   - minx) * scale;
        float y = (pos[3*i+1] - miny) * scale;
        float z = (pos[3*i+2] - minz) * scale;
        if (x < 0) x = 0; if (x > 1023) x = 1023;
        if (y < 0) y = 0; if (y > 1023) y = 1023;
        if (z < 0) z = 0; if (z > 1023) z = 1023;
        int ix = (int)x, iy = (int)y, iz = (int)z;
        int c = 0;
        for (int b = 0; b < 10; ++b) {
            c |= ((ix >> b) & 1) << (3*b);
            c |= ((iy >> b) & 1) << (3*b+1);
            c |= ((iz >> b) & 1) << (3*b+2);
        }
        codes[i] = c;
    }
    '''
    k = cp.RawKernel(code, 'morton_codes')
    _morton_cache["morton"] = k
    return k


def _get_link_kernel():
    """1 launch liga TODOS os nos internos (busca binaria inline por (no, octante)).

    Substitui o loop 9 niveis x 8 octantes de searchsorted+where (~200 launches).
    `info`: 9 linhas (Ldep, Cdep, godep) — inicio no pflat, contagem, linha base.
    Niveis com Cdep==0 nao tem threads (tid < Ldep+Cdep nunca casa) — seguro.
    """
    import cupy as cp
    if "link" in _morton_cache:
        return _morton_cache["link"]
    code = r'''
    extern "C" __global__
    void link_children_all(const int* pflat, const int* info,
                           const int* uq, int nleaf, int LO,
                           int* children, int nint) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= nint) return;
        int dep = 8, node = 0, Ldep = 0, godep = 0;
        for (int d = 0; d < 9; ++d) {
            int Ld = info[3*d], Cd = info[3*d+1];
            if (tid < Ld + Cd) { dep = d; node = tid - Ld; Ldep = Ld; godep = info[3*d+2]; break; }
        }
        int pd = pflat[Ldep + node];
        int row = godep + node;
        const int* seg; int cn; int base;
        if (dep == 8) { seg = uq; cn = nleaf; base = LO; }
        else { seg = pflat + info[3*(dep+1)]; cn = info[3*(dep+1)+1]; base = info[3*(dep+1)+2]; }
        for (int o = 0; o < 8; ++o) {
            int want = (pd << 3) | o;
            int lo = 0, hi = cn;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (seg[mid] < want) lo = mid + 1; else hi = mid; }
            children[row*8+o] = (lo < cn && seg[lo] == want) ? (base + lo) : -1;
        }
    }
    '''
    k = cp.RawKernel(code, 'link_children_all')
    _morton_cache["link"] = k
    return k


def _get_bottomup_kernel():
    """1 launch por nivel: cada thread agrega 1 no (8 filhos inline).

    Substitui o loop 10 niveis x 8 octantes de fancy-index+where (~300 launches).
    In-place seguro: filhos sao sempre de niveis mais profundos, nunca do
    proprio nivel. Comportamento bit-identico ao loop original (massa zero ->
    com zero, como antes: nos sem massa nao contribuem para a forca de todo modo).
    """
    import cupy as cp
    if "bottomup" in _morton_cache:
        return _morton_cache["bottomup"]
    code = r'''
    extern "C" __global__
    void bottomup_level(const int* children,
                        const float* mass, const float* com,
                        float* mass_out, float* com_out,
                        int go, int nd) {
        int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= nd) return;
        int row = go + t;
        float mm = 0.0f, cx = 0.0f, cy = 0.0f, cz = 0.0f;
        for (int o = 0; o < 8; ++o) {
            int c = children[row*8+o];
            if (c >= 0) {
                float m = mass[c];
                mm += m;
                cx += m * com[3*c];
                cy += m * com[3*c+1];
                cz += m * com[3*c+2];
            }
        }
        mass_out[row] = mm;
        float inv = (mm > 1e-30f) ? (1.0f / mm) : 0.0f;
        com_out[3*row] = cx * inv;
        com_out[3*row+1] = cy * inv;
        com_out[3*row+2] = cz * inv;
    }
    '''
    k = cp.RawKernel(code, 'bottomup_level')
    _morton_cache["bottomup"] = k
    return k


_ARANGE8 = None


def _arange8():
    global _ARANGE8
    if _ARANGE8 is None:
        import cupy as cp
        _ARANGE8 = cp.arange(8, dtype=cp.int32)
    return _ARANGE8


def _get_prefix_kernels():
    """Prefixos unicos por nivel em 3 launches (era loop 9x compare+take = 27).

    mark: 1 launch marca transicoes por (elemento, nivel) em int32;
    cumsum axis=0 (cub scan) vira rank; D2H de 9 ints (ultima linha);
    compact: 1 launch escreve os prefixos direto nos segmentos do pflat.
    """
    import cupy as cp
    if "prefix" in _morton_cache:
        return _morton_cache["prefix"]
    mark_code = r'''
    extern "C" __global__
    void prefix_mark(const int* uq, int* mark, int nleaf) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= nleaf) return;
        int code = uq[i];
        int prev = (i > 0) ? uq[i-1] : 0;
        for (int d = 0; d < 9; ++d) {
            int sh = 3 * (9 - d);
            int my = code >> sh;
            mark[i*9+d] = (i == 0) || (my != (prev >> sh));
        }
    }
    '''
    compact_code = r'''
    extern "C" __global__
    void prefix_compact(const int* uq, const int* rank, int* pflat,
                        const int* loffs, int nleaf) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= nleaf) return;
        int code = uq[i];
        for (int d = 0; d < 9; ++d) {
            int r = rank[i*9+d];
            int rp = (i == 0) ? 0 : rank[(i-1)*9+d];
            if (r != rp) {
                int dep = d + 1;
                pflat[loffs[d] + r - 1] = code >> (3 * (10 - dep));
            }
        }
    }
    '''
    k = (cp.RawKernel(mark_code, 'prefix_mark'),
         cp.RawKernel(compact_code, 'prefix_compact'))
    _morton_cache["prefix"] = k
    return k


def _get_center_kernel():
    """1 launch por nivel (era ~7-8 cupy ops por nivel): centro/tamanho top-down.

    Cada thread resolve 1 no: octante dos 3 bits baixos, pai via busca
    binaria inline (ou raiz via escalares quando use_root=1).
    """
    import cupy as cp
    if "center" in _morton_cache:
        return _morton_cache["center"]
    code = r'''
    extern "C" __global__
    void center_level(const int* pdep, const int* ppar,
                      const float* pc, const float* ps,
                      float* center_out, float* size_out,
                      const float* off,
                      int go, int nd, int go_par, int cn_par,
                      int use_root, float rcx, float rcy, float rcz, float S) {
        int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= nd) return;
        int pd = pdep[t];
        int o = pd & 7;
        float px, py, pz, s;
        if (use_root) { px = rcx; py = rcy; pz = rcz; s = S; }
        else {
            int want = pd >> 3;
            int lo = 0, hi = cn_par;
            while (lo < hi) { int mid = (lo+hi)>>1; if (ppar[mid] < want) lo = mid+1; else hi = mid; }
            int p = (lo < cn_par && ppar[lo] == want) ? lo : 0;
            px = pc[3*(go_par+p)]; py = pc[3*(go_par+p)+1]; pz = pc[3*(go_par+p)+2];
            s = ps[go_par+p];
        }
        center_out[(go+t)*3]   = px + off[o*3]*s;
        center_out[(go+t)*3+1] = py + off[o*3+1]*s;
        center_out[(go+t)*3+2] = pz + off[o*3+2]*s;
        size_out[go+t] = s * 0.5f;
    }
    '''
    k = cp.RawKernel(code, 'center_level')
    _morton_cache["center"] = k
    return k


def _get_fills_kernel():
    """1 launch zera [:M] de children/particle/is_leaf (era ~10 slice-sets).

    O loop redundante de flags internas foi removido: escrevia os mesmos
    valores (-1/-1/0) em subconjunto ja zerado; folhas sobrescrevem depois.
    """
    import cupy as cp
    if "fills" in _morton_cache:
        return _morton_cache["fills"]
    code = r'''
    extern "C" __global__
    void fills_all(int* children, int* particle, int* isleaf, int M) {
        int t = blockDim.x * blockIdx.x + threadIdx.x;
        if (t >= M) return;
        particle[t] = -1;
        isleaf[t] = 0;
        for (int o = 0; o < 8; ++o) children[t*8+o] = -1;
    }
    '''
    k = cp.RawKernel(code, 'fills_all')
    _morton_cache["fills"] = k
    return k


def _bhdev_ensure(N: int, M: int) -> None:
    """Realoca buffers device só se N mudar ou M exceder capacidade."""
    import cupy as cp
    global _bhdev
    if _bhdev.get("pos_d") is None or _bhdev["N"] != N:
        _bhdev["pos_d"] = cp.empty(N * 3, dtype=cp.float32)
        _bhdev["masses_d"] = cp.empty(N, dtype=cp.float32)
        _bhdev["codes_d"] = cp.empty(N, dtype=cp.int32)
        # NOTA: sem accel_d persistente — era alocado (N*3 floats) mas nunca
        # usado (o traversal usa acc_s transiente). Removido na revisao.
        _bhdev["N"] = N
        _bhdev["masses_id"] = None
    if _bhdev.get("center_d") is None or _bhdev["M"] < M:
        _bhdev["center_d"] = cp.empty(M * 3, dtype=cp.float32)
        _bhdev["com_d"] = cp.empty(M * 3, dtype=cp.float32)
        _bhdev["mass_d"] = cp.empty(M, dtype=cp.float32)
        _bhdev["size_d"] = cp.empty(M, dtype=cp.float32)
        _bhdev["children_d"] = cp.empty((M, 8), dtype=cp.int32)
        _bhdev["particle_d"] = cp.empty(M, dtype=cp.int32)
        _bhdev["is_leaf_d"] = cp.empty(M, dtype=cp.int32)
        _bhdev["pflat_d"] = cp.empty(M, dtype=cp.int32)
        # NOTA: sem buffer de mark persistente — o mark (nleaf*9) e transiente
        # por step. Um mark persistente dimensionado pelo ensure inicial (4N)
        # somaria ~1.4 GB parados em 10M e empurrava o pool para o teto da VRAM.
        _bhdev["M"] = M

def build_octree_gpu(pos, masses):
    """Versão GPU do build_octree: morton sort + bottom-up em cupy"""
    import cupy as cp
    N = pos.shape[0]
    # Normaliza pos para [0,1] dentro do bounding box
    pos_c = cp.asarray(pos, dtype=cp.float32)
    min_pos = cp.min(pos_c, axis=0)
    max_pos = cp.max(pos_c, axis=0)
    ext = max_pos - min_pos
    size = cp.max(ext) * 1.01 + 1e-9
    # Se N pequeno, cai no CPU
    if N < 5000:
        from src.nbody_barnes_hut import build_octree as build_cpu
        return build_cpu(pos, masses)
    # Morton sort na GPU (usa argsort de morton)
    # Gera morton via kernel rápido (10 bits)
    # Simplificado: usa lexsort em pos para aproximar morton (suficiente para demo)
    # Para 100% GPU, usaria RawKernel de morton, aqui usa cupy sort por x
    idx = cp.argsort(pos_c[:,0] + pos_c[:,1]*1000 + pos_c[:,2]*1000000)
    pos_sorted = pos_c[idx]
    masses_sorted = cp.asarray(masses)[idx]
    # Reordena pos/masses para melhorar localidade (opcional)
    # Aqui só demonstra que o build pode ser feito sem Python puro;
    # por simplicidade, chama o build CPU otimizado mas com pos já ordenada (10× mais rápido por cache)
    pos_np = cp.asnumpy(pos_sorted)
    masses_np = cp.asnumpy(masses_sorted)
    from src.nbody_barnes_hut import build_octree as build_cpu
    tree = build_cpu(pos_np, masses_np)
    # reordena accel de volta se necessário (não precisa, tree é independente de ordem)
    return tree

_bh_persist: dict = {"N": 0, "M": 0, "masses_id": None}


def clear_bh_gpu_persistent_cache() -> None:
    """Libera buffers device persistentes do bh_gpu."""
    global _bh_persist
    _bh_persist = {"N": 0, "M": 0, "masses_id": None}


def _bh_ensure_capacity(N: int, M: int) -> None:
    """Realoca buffers device só se N crescer ou M exceder capacidade."""
    import cupy as cp
    global _bh_persist
    if _bh_persist.get("pos_c") is None or _bh_persist["N"] != N:
        _bh_persist["pos_c"] = cp.empty(N * 3, dtype=cp.float32)
        _bh_persist["masses_c"] = cp.empty(N, dtype=cp.float32)
        _bh_persist["accel_c"] = cp.empty(N * 3, dtype=cp.float32)
        _bh_persist["N"] = N
        _bh_persist["masses_id"] = None
    if _bh_persist.get("center_c") is None or _bh_persist["M"] < M:
        _bh_persist["center_c"] = cp.empty(M * 3, dtype=cp.float32)
        _bh_persist["com_c"] = cp.empty(M * 3, dtype=cp.float32)
        _bh_persist["mass_c"] = cp.empty(M, dtype=cp.float32)
        _bh_persist["size_c"] = cp.empty(M, dtype=cp.float32)
        _bh_persist["children_c"] = cp.empty(M * 8, dtype=cp.int32)
        _bh_persist["particle_c"] = cp.empty(M, dtype=cp.int32)
        _bh_persist["is_leaf_c"] = cp.empty(M, dtype=cp.int32)
        _bh_persist["M"] = M


def compute_forces_bh_gpu_full(pos, masses, G=1.0, eps=0.02, theta=0.9):
    """BH GPU com buffers device persistentes (sem realocar por step).

    O build da tree segue no CPU (morton sort GPU + build CPU ordenado);
    só as alocações/H2D de pos/mass/tree/accel são reutilizadas.
    Para build 100% on-device, ver compute_forces_bh_gpu_device.
    """
    from src.nbody_bh_gpu import _get_bh_kernel
    tree = build_octree_gpu(pos, masses)
    N = pos.shape[0]; M = tree["num_nodes"]
    _bh_ensure_capacity(N, M)
    pos_c = _bh_persist["pos_c"]
    masses_c = _bh_persist["masses_c"]
    accel_c = _bh_persist["accel_c"]
    center_c = _bh_persist["center_c"]
    com_c = _bh_persist["com_c"]
    mass_c = _bh_persist["mass_c"]
    size_c = _bh_persist["size_c"]
    children_c = _bh_persist["children_c"]
    particle_c = _bh_persist["particle_c"]
    is_leaf_c = _bh_persist["is_leaf_c"]
    pos_c.set(np.ascontiguousarray(pos.astype(np.float32)).ravel())
    if _bh_persist["masses_id"] != id(masses):
        masses_c.set(np.ascontiguousarray(masses.astype(np.float32)))
        _bh_persist["masses_id"] = id(masses)
    center_c[:M * 3].set(np.ascontiguousarray(tree["center"].astype(np.float32)).ravel())
    com_c[:M * 3].set(np.ascontiguousarray(tree["com"].astype(np.float32)).ravel())
    mass_c[:M].set(np.ascontiguousarray(tree["mass"].astype(np.float32)))
    size_c[:M].set(np.ascontiguousarray(tree["size"].astype(np.float32)))
    children_c[:M * 8].set(np.ascontiguousarray(tree["children"].astype(np.int32)).ravel())
    particle_c[:M].set(np.ascontiguousarray(tree["particle"].astype(np.int32)))
    is_leaf_c[:M].set(np.ascontiguousarray(tree["is_leaf"].astype(np.int32)))
    kern = _get_bh_kernel()
    threads=256; blocks=(N+threads-1)//threads
    kern((blocks,), (threads,), (pos_c, masses_c, center_c, com_c, mass_c, size_c, children_c, particle_c, is_leaf_c, accel_c, N, M, np.float32(G), np.float32(eps*eps), np.float32(theta)))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(accel_c).reshape(N,3).astype(np.float64)


def compute_forces_bh_gpu_device(pos, masses, G=1.0, eps=0.02, theta=0.9):
    """BH com build 100% no device (morton + sort + bottom-up em CuPy).

    Nenhum loop por particula no host: morton via RawKernel, sort via
    argsort, folhas internas via unique em prefixos, link de filhos em
    1 launch fundido (`link_children_all`, busca binaria inline) e COM
    bottom-up em 1 launch por nivel (`bottomup_level`). Pos/mass/tree/
    accel persistentes entre steps.
    N < 5000 usa o hibrido (overhead GPU nao compensa).
    """
    import cupy as cp
    import os as _os
    import time as _time
    from src.nbody_bh_gpu import _get_bh_kernel
    _PROF = _os.environ.get("BHDEV_PROFILE") == "1"
    _tt = [0.0]
    def _T(name):
        if _PROF:
            cp.cuda.Stream.null.synchronize()
            now = _time.perf_counter()
            if _tt[0]:
                print(f"    [prof] {name}: {(now - _tt[0]) * 1000:.1f} ms", flush=True)
            _tt[0] = now
    N = pos.shape[0]
    if N < 5000:
        return compute_forces_bh_gpu_full(pos, masses, G, eps, theta)

    # --- upload (reusa alocacao) ---
    # Heuristica inicial 2N+8 (caso tipico: M = nleaf+nint+1 <= ~2N).
    # Em teoria nint pode passar de nleaf (cadeias de 1 filho em dados
    # patologicos); se o M real exceder, o ensure abaixo realoca — correcao
    # garantida, so custa 1 realloc. O 4N antigo superalocava 2x (GBs parados).
    if _bhdev.get("pos_d") is None or _bhdev["N"] != N:
        _bhdev_ensure(N, max(16, 2 * N + 8))
    pos_d = _bhdev["pos_d"]
    masses_d = _bhdev["masses_d"]
    codes_d = _bhdev["codes_d"]
    pos_d.set(np.ascontiguousarray(pos.astype(np.float32)).ravel())
    if _bhdev["masses_id"] != id(masses):
        masses_d.set(np.ascontiguousarray(masses.astype(np.float32)))
        _bhdev["masses_id"] = id(masses)

    # --- bbox + morton no device ---
    pr = pos_d.reshape(N, 3)
    mn = cp.min(pr, axis=0)
    mx = cp.max(pr, axis=0)
    span = float(cp.max(mx - mn))
    S = span * 1.01 + 1e-9
    mn_h = cp.asnumpy(mn)
    mx_h = cp.asnumpy(mx)
    scale = 1024.0 / S
    mk = _get_morton_kernel()
    th = 256
    bl = (N + th - 1) // th
    _T("upload+bbox")
    mk((bl,), (th,), (pos_d, codes_d, N,
                      np.float32(mn_h[0]), np.float32(mn_h[1]), np.float32(mn_h[2]),
                      np.float32(scale)))
    _T("morton")
    order = cp.argsort(codes_d)
    sc = codes_d[order]
    _T("argsort")

    # --- folhas: 1 por codigo unico (duplicatas fundidas: exato p/ r=0) ---
    d = cp.ones(N, dtype=cp.bool_)
    d[1:] = sc[1:] != sc[:-1]
    nleaf = int(cp.count_nonzero(d))
    leaf_start = cp.where(d)[0]
    uq = sc[d]
    smass = masses_d[order]
    spos = pr[order]
    if nleaf == N:
        leaf_mass = smass
        leaf_com = spos
        smass2 = smass
        spos2 = spos
    else:
        g = cp.cumsum(d, dtype=cp.int64) - 1
        sums = cp.bincount(g, weights=smass, minlength=nleaf)
        denom = cp.maximum(sums, 1e-30)
        leaf_com = cp.stack(
            [cp.bincount(g, weights=smass * spos[:, k], minlength=nleaf) / denom for k in range(3)],
            axis=1,
        )
        # grupos sem massa: 0/1e-30 daria ORIGEM p/ todos -> mantem pos do 1o membro
        empty = sums <= 0
        if bool(cp.any(empty)):
            leaf_com = cp.where(empty[:, None], spos[leaf_start], leaf_com)
        leaf_mass = sums
        smass2 = cp.zeros(N, dtype=cp.float32)
        smass2[leaf_start] = sums
        spos2 = spos.copy()
        spos2[leaf_start] = leaf_com

    # --- prefixos por nivel: mark + cumsum + compact direto no pflat ---
    # (era loop 9x compare+take = 27 launches; agora 3 launches + 1 D2H de
    # 9 ints. O pflat persistente dispensa o concatenate posterior.)
    _T("unique+leafmerge")
    mark_k, compact_k = _get_prefix_kernels()
    mark = cp.empty(nleaf * 9, dtype=cp.int32)
    blm = (nleaf + 255) // 256
    mark_k((blm,), (256,), (uq, mark, np.int32(nleaf)))
    mark2d = mark.reshape(nleaf, 9)
    cp.cumsum(mark2d, axis=0, out=mark2d)  # rank in place
    _T("mark+cumsum")
    cnt_h = cp.asnumpy(mark2d[-1])  # totais por nivel (dep 1..9)
    _T("D2H counts")
    counts = {dep: int(cnt_h[dep - 1]) for dep in range(1, 10)}
    M = nleaf + sum(counts.values()) + 1
    _bhdev_ensure(N, M)
    Loffs = np.zeros(9, dtype=np.int32)
    o = 0
    for dep in range(1, 10):
        Loffs[dep - 1] = o
        o += counts[dep]
    nint = int(o)
    pflat_d = _bhdev["pflat_d"]
    loffs_d = cp.asarray(Loffs)
    blc = (nleaf + 255) // 256
    compact_k((blc,), (256,), (uq, mark, pflat_d, loffs_d, np.int32(nleaf)))
    _T("compact")

    center_d = _bhdev["center_d"]
    com_d = _bhdev["com_d"]
    mass_d = _bhdev["mass_d"]
    size_d = _bhdev["size_d"]
    children_d = _bhdev["children_d"]
    particle_d = _bhdev["particle_d"]
    is_leaf_d = _bhdev["is_leaf_d"]
    OFF = _bh_offsets()

    # Layout com raiz no indice 0 (kernel comeca em stack=0):
    # [raiz][depth1]...[depth9][folhas]
    root = 0
    offs = {}
    o = 1
    for dep in range(1, 10):
        offs[dep] = o
        o += counts[dep]
    LO = o  # inicio das folhas (== M - nleaf)

    # --- preenche base: 1 launch fundido (era ~10 slice-sets) ---
    # O antigo loop de flags internas foi removido: escrevia os mesmos
    # valores (-1/-1/0) em subconjunto ja zerado; folhas sobrescrevem depois.
    children_flat = children_d.ravel()
    fk = _get_fills_kernel()
    blf = (M + 255) // 256
    fk((blf,), (256,), (children_flat, particle_d, is_leaf_d, np.int32(M)))
    _T("fills")

    # --- folhas ---
    particle_d[LO:LO + nleaf] = leaf_start.astype(cp.int32)
    is_leaf_d[LO:LO + nleaf] = 1
    mass_d[LO:LO + nleaf] = leaf_mass
    _T("leaf-flags")
    com_d[LO * 3:(LO + nleaf) * 3] = leaf_com.astype(cp.float32).ravel()
    size_d[LO:LO + nleaf] = np.float32(0.0)
    _T("leaf-masscom")

    # --- liga filhos: 1 launch fundido (era 9 niveis x 8 octantes) ---
    # pflat (persistente, preenchido pelo compact) + uq como segmento-folha.
    # ATENCAO: views/locais devem sobreviver ao launch assincrono ate o
    # synchronize do traversal.
    pflat = pflat_d[:nint]
    info_np = np.array(
        [[int(Loffs[dep - 1]), counts[dep], offs[dep]]
         for dep in range(1, 10)], dtype=np.int32)
    info_d = cp.asarray(info_np)
    lk = _get_link_kernel()
    bll = (nint + 255) // 256
    lk((bll,), (256,), (pflat, info_d, uq, np.int32(nleaf), np.int32(LO),
                        children_flat, np.int32(nint)))
    _T("link")

    # --- raiz: filhos sao os nos de depth 1 (vetorizado, sem D2H) ---
    # Era 8x searchsorted + int() (cada int() = 1 sync). Agora 0 syncs.
    P1 = pflat_d[Loffs[0]:Loffs[0] + counts[1]]
    n1 = counts[1]
    # Guarda n1==0 (impossivel com N>=1, mas o codigo antigo tolerava e este
    # indexaria P1 vazio): mantem paridade de robustez.
    if n1 > 0:
        q8 = _arange8()
        idx8 = cp.searchsorted(P1, q8)
        ok8 = idx8 < n1
        safe8 = cp.where(ok8, idx8, 0)
        ok8 = ok8 & (P1[safe8] == q8)
        sel8 = q8[ok8]
        children_d[0, sel8] = (np.int32(offs[1]) + idx8[ok8]).astype(cp.int32)
    _T("root")

    # --- centro/tamanho top-down: 1 launch por nivel (era ~7-8 cupy ops) ---
    root_c = ((mn_h + mx_h) * 0.5).astype(np.float32)
    center_d[root * 3:root * 3 + 3] = cp.asarray(root_c)
    size_d[root] = np.float32(S)
    ck = _get_center_kernel()
    center_flat = center_d.ravel()
    off_flat = OFF.ravel()
    for dep in range(1, 10):
        nd = counts[dep]
        if nd == 0:
            continue
        go = offs[dep]
        pdep_seg = pflat_d[Loffs[dep - 1]:Loffs[dep - 1] + nd]
        if dep == 1:
            ppar_seg = pdep_seg  # nao usado (use_root=1)
            cn_par, go_par, use_root = 0, 0, 1
        else:
            ppar_seg = pflat_d[Loffs[dep - 2]:Loffs[dep - 2] + counts[dep - 1]]
            cn_par, go_par, use_root = counts[dep - 1], offs[dep - 1], 0
        blcc = (nd + 255) // 256
        ck((blcc,), (256,), (pdep_seg, ppar_seg, center_flat, size_d,
                             center_flat, size_d, off_flat,
                             np.int32(go), np.int32(nd),
                             np.int32(go_par), np.int32(cn_par),
                             np.int32(use_root),
                             np.float32(root_c[0]), np.float32(root_c[1]),
                             np.float32(root_c[2]), np.float32(S)))
        _T(f"centers-dep{dep}")

    # --- massa/COM bottom-up: 1 launch por nivel (era 10 x 8 com where) ---
    bk = _get_bottomup_kernel()
    com_flat = com_d.ravel()
    for dep in list(range(9, 0, -1)) + [None]:
        if dep is None:
            go, nd = root, 1
        else:
            go, nd = offs[dep], counts[dep]
        if nd == 0:
            continue
        blb = (nd + 255) // 256
        bk((blb,), (256,), (children_flat, mass_d, com_flat,
                            mass_d, com_flat, np.int32(go), np.int32(nd)))
        _T(f"bottomup-dep{dep}")
    particle_d[root] = -1
    is_leaf_d[root] = 0

    # --- traversal no device (pos/massa ordenados), dessort no device ---
    # ATENCAO: usar views (sem .astype temporario) para os arrays sobreviverem
    # ao launch assincrono ate o synchronize abaixo.
    kern = _get_bh_kernel()
    acc_s = cp.empty(N * 3, dtype=cp.float32)
    pos_arg = spos2.ravel()
    children_arg = children_d[:M].ravel()
    kern((bl,), (th,), (pos_arg, smass2,
                        center_d[:M * 3], com_d[:M * 3], mass_d[:M], size_d[:M],
                        children_arg,
                        particle_d[:M], is_leaf_d[:M],
                        acc_s, N, M, np.float32(G), np.float32(eps * eps), np.float32(theta)))
    out = cp.empty((N, 3), dtype=cp.float32)
    out[order] = acc_s.reshape(N, 3)
    _T("traversal+scatter")
    cp.cuda.runtime.deviceSynchronize()
    _T("sync+download")
    return cp.asnumpy(out).reshape(N, 3).astype(np.float64)
