import cupy as cp
import numpy as np
from typing import Tuple, Optional
import time

# CUDA kernel for Monte-Carlo CFR rollouts
ROLLOUT_KERNEL = """
#include <cstdint>
extern "C" __global__
void rollout_kernel(
    const unsigned long long* __restrict__ keys,
    float* __restrict__ cf_values,
    const unsigned long long seed,
    const int batch_size,
    const int num_actions,
    const int N_rollouts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (num_actions * N_rollouts);
    int action_idx = (idx % (num_actions * N_rollouts)) / N_rollouts;
    int rollout_idx = idx % N_rollouts;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize random state
    unsigned long long state = keys[batch_idx * num_actions + action_idx] + seed + rollout_idx;
    
    // Simple poker simulation parameters
    const int stack_size = 10000;  // 100 BB
    const int pot_size = 150;      // 1.5 BB
    const int max_depth = 4;       // Preflop -> River
    
    // Simulate poker hand
    float payoff = 0.0f;
    
    // Use the key to seed the simulation
    for (int depth = 0; depth < max_depth; depth++) {
        // Simple action selection based on key
        int action = (state >> (depth * 8)) % 4;  // 0=fold, 1=call, 2=bet, 3=raise
        
        // Update state for next iteration
        state = state * 1103515245ULL + 12345ULL;
        
        // Simple payoff calculation
        if (action == 0) {  // fold
            payoff = -pot_size * 0.5f;
            break;
        } else if (action == 1) {  // call
            payoff = (state % 200 - 100) * 0.1f;  // Random outcome
        } else if (action == 2) {  // bet
            payoff = (state % 300 - 150) * 0.15f;
        } else {  // raise
            payoff = (state % 400 - 200) * 0.2f;
        }
    }
    
    // Store result
    if (rollout_idx < N_rollouts) {
        atomicAdd(&cf_values[batch_idx * num_actions + action_idx], payoff);
    }
}

extern "C" __global__
void normalize_rollouts_kernel(
    float* cf_values,
    const int batch_size,
    const int num_actions,
    const int N_rollouts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_actions;
    int action_idx = idx % num_actions;
    
    if (batch_idx >= batch_size) return;
    
    // Normalize by number of rollouts
    int value_idx = batch_idx * num_actions + action_idx;
    cf_values[value_idx] = cf_values[value_idx] / N_rollouts;
}
"""

# Compile the kernel
ROLLOUT_MODULE = cp.RawModule(code=ROLLOUT_KERNEL)
rollout_kernel = ROLLOUT_MODULE.get_function("rollout_kernel")
normalize_kernel = ROLLOUT_MODULE.get_function("normalize_rollouts_kernel")

def mccfr_rollout_gpu(
    keys_gpu: cp.ndarray,
    table_keys: cp.ndarray,
    table_vals: cp.ndarray,
    counter: cp.ndarray,
    N_rollouts: int = 100
) -> cp.ndarray:
    """
    Monte-Carlo CFR rollout on GPU.
    
    Args:
        keys_gpu: (B, num_actions) uint64 keys for each action
        table_keys: GPU hash table keys (persistent)
        table_vals: GPU hash table values (persistent)
        counter: GPU counter for unique keys (persistent)
        N_rollouts: Number of rollouts per action (default: 100)
    
    Returns:
        cf_values: (B, num_actions) counterfactual values as float32 ready for scatter_update
    """
    batch_size, num_actions = keys_gpu.shape
    
    # Allocate output array
    cf_values = cp.zeros((batch_size, num_actions), dtype=cp.float32)
    
    # Calculate grid and block dimensions
    total_threads = batch_size * num_actions * N_rollouts
    threads_per_block = 256
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
    
    # Launch rollout kernel
    rollout_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            keys_gpu,
            cf_values,
            cp.uint64(int(time.time() * 1000)),  # Seed
            batch_size,
            num_actions,
            N_rollouts
        )
    )
    
    # Normalize results
    normalize_blocks = (batch_size * num_actions + 255) // 256
    normalize_kernel(
        (normalize_blocks,),
        (256,),
        (cf_values, batch_size, num_actions, N_rollouts)
    )
    
    return cf_values

def benchmark_mccfr_rollout():
    """Benchmark the MCCFR rollout performance."""
    print("Benchmarking MCCFR rollout...")
    
    # Test parameters
    batch_size = 1024
    num_actions = 6
    N_rollouts = 100
    
    # Create dummy keys
    keys_gpu = cp.random.randint(0, 2**32, (batch_size, num_actions), dtype=cp.uint64)
    
    # Create dummy hash table arrays
    table_keys = cp.zeros(2**24, dtype=cp.uint64)
    table_vals = cp.zeros(2**24, dtype=cp.uint64)
    counter = cp.zeros(1, dtype=cp.uint64)
    
    # Warm up
    for _ in range(3):
        _ = mccfr_rollout_gpu(keys_gpu, table_keys, table_vals, counter, N_rollouts)
    
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start_time = time.time()
    cf_values = mccfr_rollout_gpu(keys_gpu, table_keys, table_vals, counter, N_rollouts)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    total_rollouts = batch_size * num_actions * N_rollouts
    throughput = total_rollouts / (end_time - start_time)
    
    print(f"Batch size: {batch_size}")
    print(f"Actions per batch: {num_actions}")
    print(f"Rollouts per action: {N_rollouts}")
    print(f"Total rollouts: {total_rollouts:,}")
    print(f"Time: {end_time - start_time:.3f}s")
    print(f"Throughput: {throughput:,.0f} rollouts/sec")
    print(f"Memory usage: ~{batch_size * num_actions * 8 / 1024 / 1024:.1f} MB")
    
    return cf_values

if __name__ == "__main__":
    # Run benchmark
    cf_values = benchmark_mccfr_rollout()
    print(f"Sample cf_values shape: {cf_values.shape}")
    print(f"Sample values: {cf_values[:5, :5]}") 