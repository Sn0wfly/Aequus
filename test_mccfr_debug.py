#!/usr/bin/env python3
"""
Debug script for MCCFR GPU kernel to isolate illegal memory access issue.
"""

import cupy as cp
import numpy as np
import sys
import os

# Add the poker_bot directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'poker_bot', 'core'))

from mccfr_gpu import mccfr_rollout_gpu

def test_small_batch():
    """Test with very small batch to isolate the issue."""
    print("🧪 Testing MCCFR kernel with small batch...")
    
    # Very small test
    batch_size = 10
    num_actions = 14
    N_rollouts = 10
    
    # Create small keys array
    keys_gpu = cp.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=cp.uint64)
    
    print(f"Test parameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_actions: {num_actions}")
    print(f"  N_rollouts: {N_rollouts}")
    print(f"  keys shape: {keys_gpu.shape}")
    print(f"  keys dtype: {keys_gpu.dtype}")
    
    try:
        # Call the function
        cf_values = mccfr_rollout_gpu(keys_gpu, N_rollouts=N_rollouts, num_actions=num_actions)
        
        print(f"✅ SUCCESS! cf_values shape: {cf_values.shape}")
        print(f"Sample values: {cf_values[:3, :5]}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_medium_batch():
    """Test with medium batch size."""
    print("\n🧪 Testing MCCFR kernel with medium batch...")
    
    batch_size = 100
    num_actions = 14
    N_rollouts = 50
    
    # Create medium keys array
    keys_gpu = cp.random.randint(0, 10000, (batch_size,), dtype=cp.uint64)
    
    print(f"Test parameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_actions: {num_actions}")
    print(f"  N_rollouts: {N_rollouts}")
    print(f"  keys shape: {keys_gpu.shape}")
    
    try:
        cf_values = mccfr_rollout_gpu(keys_gpu, N_rollouts=N_rollouts, num_actions=num_actions)
        print(f"✅ SUCCESS! cf_values shape: {cf_values.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_large_batch():
    """Test with large batch size (like the failing case)."""
    print("\n🧪 Testing MCCFR kernel with large batch...")
    
    batch_size = 512  # Same as debug config
    num_actions = 14
    N_rollouts = 100
    
    # Create large keys array
    keys_gpu = cp.random.randint(0, 20000, (batch_size,), dtype=cp.uint64)
    
    print(f"Test parameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_actions: {num_actions}")
    print(f"  N_rollouts: {N_rollouts}")
    print(f"  keys shape: {keys_gpu.shape}")
    
    try:
        cf_values = mccfr_rollout_gpu(keys_gpu, N_rollouts=N_rollouts, num_actions=num_actions)
        print(f"✅ SUCCESS! cf_values shape: {cf_values.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 MCCFR GPU Kernel Debug Test")
    print("=" * 50)
    
    # Test with increasing batch sizes
    test_small_batch()
    test_medium_batch()
    test_large_batch()
    
    print("\n🎉 Debug test completed!") 