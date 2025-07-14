"""
GPU-only bucketing + hash-table for Poker CFR
Requires: cupy-cuda12x (or your CUDA version)
"""

import cupy as cp
import time

# -----------------------------
# 1. Empaquetado de claves
# -----------------------------
def pack_keys(hole_hash, round_id, position,
              stack_bucket, pot_bucket, num_active):
    """
    Todos los inputs son CuPy arrays del mismo shape
    Devuelve CuPy array uint64
    """
    return (
        (hole_hash.astype(cp.uint64) << 24) |
        (round_id.astype(cp.uint64) << 16) |
        (position.astype(cp.uint64) << 8) |
        (stack_bucket.astype(cp.uint64) << 4) |
        (pot_bucket.astype(cp.uint64) << 1) |
        (num_active.astype(cp.uint64))
    )

# -----------------------------
# 2. Hash-table optimizada en GPU
# -----------------------------
_HASH_KERNEL = r'''
extern "C" __global__
void build_or_get_indices(
    const unsigned long long* __restrict__ keys,
    unsigned int* __restrict__ indices,
    unsigned long long* __restrict__ table_keys,
    unsigned int* __restrict__ table_vals,
    const unsigned int N,
    const unsigned int table_size)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    unsigned long long key = keys[tid];
    // OPTIMIZACIÓN 1: Bitwise AND en vez de módulo (más rápido)
    unsigned int slot = (unsigned int)(key & (table_size - 1));

    while (true) {
        unsigned long long old = atomicCAS(&table_keys[slot], 0ULL, key);
        if (old == 0ULL || old == key) {
            // key inserted or found
            if (old == 0ULL) {
                // new key: claim next index
                unsigned int idx = atomicAdd(&table_vals[slot], 1u);
                indices[tid] = idx;
            } else {
                // existing key: read index
                indices[tid] = table_vals[slot];
            }
            break;
        }
        // OPTIMIZACIÓN 2: Linear probing más eficiente
        slot = (slot + 1) & (table_size - 1);
    }
}
'''

# Compilar kernel una sola vez
_build_or_get_indices_kernel = cp.RawKernel(_HASH_KERNEL, 'build_or_get_indices')

# -----------------------------
# 3. Wrapper Python optimizado
# -----------------------------
def build_or_get_indices(keys_gpu, table_size=2**24):  # OPTIMIZACIÓN 3: Tabla más grande
    """
    keys_gpu: CuPy array uint64
    Devuelve: indices_gpu CuPy array uint32
    """
    N = keys_gpu.size
    indices_gpu = cp.empty(N, dtype=cp.uint32)

    # Tablas device
    table_keys = cp.zeros(table_size, dtype=cp.uint64)  # 0 == empty
    table_vals = cp.zeros(table_size, dtype=cp.uint32)  # índice

    # OPTIMIZACIÓN 4: Más threads por bloque para mejor occupancy
    threads = 512  # Aumentado de 256 a 512
    blocks = (N + threads - 1) // threads
    _build_or_get_indices_kernel(
        (blocks,), (threads,),
        (keys_gpu, indices_gpu, table_keys, table_vals,
         cp.uint32(N), cp.uint32(table_size))
    )
    cp.cuda.Device().synchronize()
    return indices_gpu

# -----------------------------
# 4. Benchmark
# -----------------------------
def benchmark():
    N = 1_000_000
    print(f"Benchmarking {N:,} keys …")

    # Datos sintéticos
    rng = cp.random.default_rng(42)
    keys = pack_keys(
        hole_hash   = rng.integers(0, 1326,  N, dtype=cp.uint16),  # 52 choose 2
        round_id    = rng.integers(0, 6,     N, dtype=cp.uint8),   # 0,3,4,5
        position    = rng.integers(0, 6,     N, dtype=cp.uint8),
        stack_bucket= rng.integers(0, 16,    N, dtype=cp.uint8),
        pot_bucket  = rng.integers(0, 16,    N, dtype=cp.uint8),
        num_active  = rng.integers(2, 7,     N, dtype=cp.uint8)
    )

    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    indices = build_or_get_indices(keys)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"GPU throughput: {N/elapsed*1e-6:.1f} M keys/sec (tiempo: {elapsed:.4f} s)")

if __name__ == '__main__':
    benchmark() 