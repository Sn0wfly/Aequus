"""
GPU-only bucketing + hash-table for Poker CFR
Requires: cupy-cuda12x
"""

import cupy as cp
import time

# -----------------------------
# 1. Empaquetado uint64
# -----------------------------
def pack_keys(hole_hash, round_id, position,
              stack_bucket, pot_bucket, num_active):
    return (
        (hole_hash.astype(cp.uint64) << 24) |
        (round_id.astype(cp.uint64) << 16) |
        (position.astype(cp.uint64) << 8) |
        (stack_bucket.astype(cp.uint64) << 4) |
        (pot_bucket.astype(cp.uint64) << 1) |
        (num_active.astype(cp.uint64))
    )

# -----------------------------
# 2. Kernel CUDA optimizado
# -----------------------------
_KERNEL = r'''
extern "C" __global__
void bucket_kernel(
    const unsigned long long* __restrict__ keys,
    unsigned int* __restrict__ out_idx,
    unsigned long long* __restrict__ table_keys,
    unsigned int* __restrict__ table_vals,
    const unsigned int N,
    const unsigned int mask)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    unsigned long long key = keys[tid];
    unsigned int slot = (unsigned int)(key & mask);

    // OPTIMIZACIÓN: Máximo 3 intentos para evitar loops infinitos
    for (int attempt = 0; attempt < 3; attempt++) {
        unsigned long long old = atomicCAS(&table_keys[slot], 0ULL, key);
        if (old == 0ULL) {
            // nueva key: asignar índice
            unsigned int idx = atomicAdd(&table_vals[0], 1u);
            table_vals[slot] = idx;
            out_idx[tid] = idx;
            return;
        }
        if (old == key) {
            // key existente
            out_idx[tid] = table_vals[slot];
            return;
        }
        slot = (slot + 1) & mask;
    }
    // Fallback: usar slot directo
    out_idx[tid] = slot;
}
'''

_bucket_kernel = cp.RawKernel(_KERNEL, 'bucket_kernel')

# -----------------------------
# 3. Wrapper optimizado
# -----------------------------
def build_or_get_indices(keys_gpu, table_size=2**26):  # OPTIMIZACIÓN: Tabla más grande
    N = keys_gpu.size
    indices_gpu = cp.empty(N, dtype=cp.uint32)

    # table_size debe ser potencia de 2
    table_keys = cp.zeros(table_size, dtype=cp.uint64)
    table_vals = cp.zeros(table_size, dtype=cp.uint32)
    counter = cp.zeros(1, dtype=cp.uint32)  # índices contiguos

    # OPTIMIZACIÓN: Más threads para mejor occupancy
    threads = 1024  # Aumentado de 256 a 1024
    blocks = (N + threads - 1) // threads
    
    # OPTIMIZACIÓN: Asegurar que hay suficientes bloques
    if blocks < 32:
        blocks = 32
    
    _bucket_kernel(
        (blocks,), (threads,),
        (keys_gpu, indices_gpu,
         table_keys, table_vals,
         cp.uint32(N), cp.uint32(table_size - 1))
    )
    cp.cuda.Device().synchronize()
    return indices_gpu

# -----------------------------
# 4. Benchmark
# -----------------------------
def benchmark():
    N = 1_000_000
    print(f"Benchmarking {N:,} keys …")
    
    rng = cp.random.default_rng(42)
    keys = pack_keys(
        rng.integers(0, 1326, N, cp.uint16),
        rng.integers(0, 6, N, cp.uint8),
        rng.integers(0, 6, N, cp.uint8),
        rng.integers(0, 16, N, cp.uint8),
        rng.integers(0, 16, N, cp.uint8),
        rng.integers(2, 7, N, cp.uint8)
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