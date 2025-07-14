#!/usr/bin/env python3
"""
Debug script to test the counter and hash table lookup.
"""

import jax
import jax.numpy as jnp
import cupy as cp
import numpy as np
from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.core.bucket_gpu import build_or_get_indices

def test_counter_debug():
    """Test the counter and hash table lookup directly."""
    print("🔍 Testing counter and hash table lookup...")
    
    # Create trainer to access hash table
    trainer = PokerTrainer(TrainerConfig(batch_size=4))
    
    # Create obviously different keys
    keys = cp.array([
        0x1004501500000000,  # Game 0, Player 0
        0x1014d01500000000,  # Game 0, Player 1
        0x1025501500000000,  # Game 0, Player 2
        0x100dd01500000000,  # Game 0, Player 3
        0x20147828c0000000,  # Game 1, Player 0
        0x2024f828c0000000,  # Game 1, Player 1
        0x200d7828c0000000,  # Game 1, Player 2
        0x201df828c0000000,  # Game 1, Player 3
    ], dtype=cp.uint64)
    
    print("📊 Test keys:")
    for i, key in enumerate(cp.asnumpy(keys)):
        print(f"   Key {i}: {key} (0x{key:016x})")
    
    # Test lookup/insertion
    print("\n🔧 Testing hash table lookup...")
    indices = build_or_get_indices(
        keys,
        trainer.table_keys,
        trainer.table_vals,
        trainer.counter
    )
    
    indices_cpu = cp.asnumpy(indices)
    counter_cpu = cp.asnumpy(trainer.counter)[0]
    
    print(f"📋 Results:")
    print(f"   Indices: {indices_cpu}")
    print(f"   Counter: {counter_cpu}")
    print(f"   Unique indices: {len(np.unique(indices_cpu))}")
    
    # Check hash table state
    print(f"\n🔍 Hash table state:")
    print(f"   Table size: {len(trainer.table_keys)}")
    print(f"   First 10 table keys: {cp.asnumpy(trainer.table_keys[:10])}")
    print(f"   First 10 table vals: {cp.asnumpy(trainer.table_vals[:10])}")
    
    # Test with more keys
    print(f"\n🔄 Testing with more keys...")
    more_keys = cp.array([
        0x300c9c3c80000000,  # Game 2, Player 0
        0x301d9c3c80000000,  # Game 2, Player 1
        0x30269c3c80000000,  # Game 2, Player 2
        0x30059c3c80000000,  # Game 2, Player 3
    ], dtype=cp.uint64)
    
    more_indices = build_or_get_indices(
        more_keys,
        trainer.table_keys,
        trainer.table_vals,
        trainer.counter
    )
    
    more_indices_cpu = cp.asnumpy(more_indices)
    new_counter_cpu = cp.asnumpy(trainer.counter)[0]
    
    print(f"   More indices: {more_indices_cpu}")
    print(f"   New counter: {new_counter_cpu}")
    print(f"   Counter increased: {new_counter_cpu - counter_cpu}")
    
    # Test with duplicate keys
    print(f"\n🔄 Testing with duplicate keys...")
    duplicate_keys = cp.array([
        0x1004501500000000,  # Same as first key
        0x1014d01500000000,  # Same as second key
        0x9999999999999999,  # New key
        0x8888888888888888,  # New key
    ], dtype=cp.uint64)
    
    dup_indices = build_or_get_indices(
        duplicate_keys,
        trainer.table_keys,
        trainer.table_vals,
        trainer.counter
    )
    
    dup_indices_cpu = cp.asnumpy(dup_indices)
    final_counter_cpu = cp.asnumpy(trainer.counter)[0]
    
    print(f"   Duplicate indices: {dup_indices_cpu}")
    print(f"   Final counter: {final_counter_cpu}")
    print(f"   Counter increase: {final_counter_cpu - new_counter_cpu}")

if __name__ == "__main__":
    test_counter_debug() 