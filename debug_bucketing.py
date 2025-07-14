#!/usr/bin/env python3
"""
Debug script to understand why bucketing generates same key for all elements.
"""

import jax
import jax.numpy as jnp
import cupy as cp
import numpy as np
from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.core.bucket_fine import fine_bucket_kernel_wrapper

def debug_bucketing():
    """Debug the bucketing process step by step."""
    print("🔍 Debugging bucketing process...")
    
    # Create test data with known differences
    batch_size = 4
    num_players = 6
    
    # Create obviously different data
    hole_cards = np.array([
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],  # Game 1
        [[12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],  # Game 2
        [[24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35]],  # Game 3
        [[36, 37], [38, 39], [40, 41], [42, 43], [44, 45], [46, 47]]   # Game 4
    ], dtype=np.int32)
    
    community_cards = np.array([
        [[0, 1, 2, -1, -1], [0, 1, 2, -1, -1], [0, 1, 2, -1, -1], [0, 1, 2, -1, -1], [0, 1, 2, -1, -1], [0, 1, 2, -1, -1]],  # Flop
        [[3, 4, 5, 6, -1], [3, 4, 5, 6, -1], [3, 4, 5, 6, -1], [3, 4, 5, 6, -1], [3, 4, 5, 6, -1], [3, 4, 5, 6, -1]],  # Turn
        [[7, 8, 9, 10, 11], [7, 8, 9, 10, 11], [7, 8, 9, 10, 11], [7, 8, 9, 10, 11], [7, 8, 9, 10, 11], [7, 8, 9, 10, 11]],  # River
        [[12, 13, 14, 15, 16], [12, 13, 14, 15, 16], [12, 13, 14, 15, 16], [12, 13, 14, 15, 16], [12, 13, 14, 15, 16], [12, 13, 14, 15, 16]]  # River
    ], dtype=np.int32)
    
    positions = np.array([
        [0, 1, 2, 3, 4, 5],  # Game 1
        [0, 1, 2, 3, 4, 5],  # Game 2
        [0, 1, 2, 3, 4, 5],  # Game 3
        [0, 1, 2, 3, 4, 5]   # Game 4
    ], dtype=np.int32)
    
    stack_sizes = np.array([
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # Game 1
        [150.0, 150.0, 150.0, 150.0, 150.0, 150.0],  # Game 2
        [200.0, 200.0, 200.0, 200.0, 200.0, 200.0],  # Game 3
        [250.0, 250.0, 250.0, 250.0, 250.0, 250.0]   # Game 4
    ], dtype=np.float32)
    
    pot_sizes = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # Game 1
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],  # Game 2
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],  # Game 3
        [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]   # Game 4
    ], dtype=np.float32)
    
    num_active = np.array([
        [6, 6, 6, 6, 6, 6],  # Game 1
        [5, 5, 5, 5, 5, 5],  # Game 2
        [4, 4, 4, 4, 4, 4],  # Game 3
        [3, 3, 3, 3, 3, 3]   # Game 4
    ], dtype=np.int32)
    
    print("📊 Input data summary:")
    print(f"   Hole cards shape: {hole_cards.shape}")
    print(f"   Community cards shape: {community_cards.shape}")
    print(f"   Positions: {positions}")
    print(f"   Stack sizes: {stack_sizes}")
    print(f"   Pot sizes: {pot_sizes}")
    print(f"   Num active: {num_active}")
    
    # Convert to CuPy
    hole_cards_gpu = cp.asarray(hole_cards, dtype=cp.int32)
    community_cards_gpu = cp.asarray(community_cards, dtype=cp.int32)
    positions_gpu = cp.asarray(positions, dtype=cp.int32)
    stack_sizes_gpu = cp.asarray(stack_sizes, dtype=cp.float32)
    pot_sizes_gpu = cp.asarray(pot_sizes, dtype=cp.float32)
    num_active_gpu = cp.asarray(num_active, dtype=cp.int32)
    
    # Call fine bucketing kernel
    print("\n🔧 Calling fine bucketing kernel...")
    keys_gpu = fine_bucket_kernel_wrapper(
        hole_cards_gpu,
        community_cards_gpu,
        positions_gpu,
        stack_sizes_gpu,
        pot_sizes_gpu,
        num_active_gpu
    )
    
    # Convert back to CPU for analysis
    keys_cpu = cp.asnumpy(keys_gpu)
    
    print("\n📋 Bucketing results:")
    print(f"   Keys shape: {keys_cpu.shape}")
    print(f"   Keys: {keys_cpu}")
    print(f"   Unique keys: {np.unique(keys_cpu)}")
    print(f"   Number of unique keys: {len(np.unique(keys_cpu))}")
    
    # Analyze each component
    print("\n🔍 Analyzing key components:")
    for i in range(batch_size):
        for j in range(num_players):
            key = keys_cpu[i, j]
            print(f"   Game {i}, Player {j}: {key} (0x{key:016x})")
            
            # Decode key components
            round_bucket = (key >> 60) & 0xF
            hand_bucket = (key >> 50) & 0x3FF
            pos_bucket = (key >> 47) & 0x7
            stack_bucket = (key >> 42) & 0x1F
            pot_bucket = (key >> 33) & 0x1FF
            active_bucket = (key >> 30) & 0x7
            hist_bucket = (key >> 18) & 0xFFF
            
            print(f"     Round: {round_bucket}, Hand: {hand_bucket}, Pos: {pos_bucket}")
            print(f"     Stack: {stack_bucket}, Pot: {pot_bucket}, Active: {active_bucket}, Hist: {hist_bucket}")

if __name__ == "__main__":
    debug_bucketing() 