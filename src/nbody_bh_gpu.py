"""
Barnes-Hut na GPU — traversal em CUDA (cupy RawKernel)
Build da octree ainda na CPU (Python), traversal paralelo em 4352 cores.
"""
import numpy as np
import numpy.typing as npt
import cupy as cp

# cache do kernel
_kernel_cache = {}

def _get_bh_kernel():
    if "bh" in _kernel_cache:
        return _kernel_cache["bh"]
    code = r'''
    extern "C" __global__
    void bh_forces(
        const float* pos,           // (N*3)
        const float* masses,        // (N)
        const float* nodes_center,  // (M*3)
        const float* nodes_com,     // (M*3)
        const float* nodes_mass,    // (M)
        const float* nodes_size,    // (M)
        const int*   nodes_children,// (M*8)
        const int*   nodes_particle,// (M)
        const int*   nodes_is_leaf, // (M) 0/1
        float* accel,               // (N*3)
        int N, int M,
        float G, float eps2, float theta
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;
        float xi = pos[3*i];
        float yi = pos[3*i+1];
        float zi = pos[3*i+2];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        // stack fixo 128 (suficiente para profundidade 16)
        int stack[128];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            int node = stack[--sp];
            int is_leaf = nodes_is_leaf[node];
            if (is_leaf) {
                int p = nodes_particle[node];
                if (p == -1 || p == i) continue;
                float dx = pos[3*p]   - xi;
                float dy = pos[3*p+1] - yi;
                float dz = pos[3*p+2] - zi;
                float r2 = dx*dx + dy*dy + dz*dz + eps2;
                float inv_r = rsqrtf(r2);
                float inv_r3 = inv_r*inv_r*inv_r;
                float s = G * masses[p] * inv_r3;
                ax += s*dx; ay += s*dy; az += s*dz;
            } else {
                float dx = nodes_com[3*node]   - xi;
                float dy = nodes_com[3*node+1] - yi;
                float dz = nodes_com[3*node+2] - zi;
                float r2 = dx*dx + dy*dy + dz*dz + eps2;
                float r = sqrtf(r2);
                float s = nodes_size[node];
                if (r < 1e-6f) r = 1e-6f;
                if (s / r < theta) {
                    // inside check: se partícula dentro do nó, não aproxima
                    float cx = nodes_center[3*node];
                    float cy = nodes_center[3*node+1];
                    float cz = nodes_center[3*node+2];
                    float half = s * 0.5f;
                    bool inside = (fabsf(xi - cx) <= half+1e-6f) && (fabsf(yi - cy) <= half+1e-6f) && (fabsf(zi - cz) <= half+1e-6f);
                    if (inside) {
                        for (int c=7; c>=0; --c) {
                            int child = nodes_children[node*8 + c];
                            if (child != -1 && sp < 128) stack[sp++] = child;
                        }
                        continue;
                    }
                    float inv_r = rsqrtf(r2);
                    float inv_r3 = inv_r*inv_r*inv_r;
                    float sf = G * nodes_mass[node] * inv_r3;
                    ax += sf*dx; ay += sf*dy; az += sf*dz;
                } else {
                    for (int c=7; c>=0; --c) {
                        int child = nodes_children[node*8 + c];
                        if (child != -1 && sp < 128) stack[sp++] = child;
                    }
                }
            }
        }
        accel[3*i]   = ax;
        accel[3*i+1] = ay;
        accel[3*i+2] = az;
    }
    '''
    kern = cp.RawKernel(code, 'bh_forces')
    _kernel_cache["bh"] = kern
    return kern

def compute_forces_bh_gpu(pos: np.ndarray, masses: np.ndarray, G=1.0, eps=0.02, theta=0.9):
    """BH GPU: build CPU + traversal GPU"""
    from src.nbody_barnes_hut import build_octree
    tree = build_octree(pos, masses)
    N = pos.shape[0]
    M = tree["num_nodes"]
    # float32
    pos_f = np.ascontiguousarray(pos.astype(np.float32))
    masses_f = np.ascontiguousarray(masses.astype(np.float32))
    nodes_center_f = np.ascontiguousarray(tree["center"].astype(np.float32))
    nodes_com_f = np.ascontiguousarray(tree["com"].astype(np.float32))
    nodes_mass_f = np.ascontiguousarray(tree["mass"].astype(np.float32))
    nodes_size_f = np.ascontiguousarray(tree["size"].astype(np.float32))
    nodes_children_i = np.ascontiguousarray(tree["children"].astype(np.int32))
    nodes_particle_i = np.ascontiguousarray(tree["particle"].astype(np.int32))
    nodes_is_leaf_i = np.ascontiguousarray(tree["is_leaf"].astype(np.int32))

    pos_c = cp.asarray(pos_f.ravel())
    masses_c = cp.asarray(masses_f)
    center_c = cp.asarray(nodes_center_f.ravel())
    com_c = cp.asarray(nodes_com_f.ravel())
    mass_c = cp.asarray(nodes_mass_f)
    size_c = cp.asarray(nodes_size_f)
    children_c = cp.asarray(nodes_children_i.ravel())
    particle_c = cp.asarray(nodes_particle_i)
    is_leaf_c = cp.asarray(nodes_is_leaf_i)
    accel_c = cp.zeros(N*3, dtype=cp.float32)

    kern = _get_bh_kernel()
    threads = 256
    blocks = (N + threads - 1)//threads
    kern((blocks,), (threads,), (
        pos_c, masses_c, center_c, com_c, mass_c, size_c, children_c, particle_c, is_leaf_c,
        accel_c, N, M, np.float32(G), np.float32(eps*eps), np.float32(theta)
    ))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(accel_c).reshape(N,3).astype(np.float64)
