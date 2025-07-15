#!/usr/bin/env python3
"""
Test script for Pluribus-style bucketing.
Compares ultra-aggressive bucketing vs fine bucketing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.core.simulation import batch_simulate_real_holdem
import jax.random as jr
import cupy as cp
import numpy as np

def test_pluribus_bucketing():
    """Test Pluribus bucketing vs fine bucketing"""
    print("🧪 Testing Pluribus Bucketing vs Fine Bucketing")
    print("=" * 60)
    
    # Test configuration
    batch_size = 1024
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    # Generate same game data for fair comparison
    rng_key = jr.PRNGKey(42)
    rng_keys = jr.split(rng_key, batch_size)
    game_results = batch_simulate_real_holdem(rng_keys, game_config)
    
    # Test 1: Fine bucketing (original)
    print("\n🔍 Test 1: Fine Bucketing (Original)")
    config_fine = TrainerConfig(
        batch_size=batch_size,
        use_pluribus_bucketing=False
    )
    trainer_fine = PokerTrainer(config_fine)
    
    results_fine = trainer_fine.train_step(game_results)
    print(f"   Unique info sets: {results_fine['unique_info_sets']:,}")
    print(f"   Info sets processed: {results_fine['info_sets_processed']:,}")
    
    # Test 2: Pluribus bucketing (ultra-aggressive)
    print("\n🔍 Test 2: Pluribus Bucketing (Ultra-Aggressive)")
    config_pluribus = TrainerConfig(
        batch_size=batch_size,
        use_pluribus_bucketing=True
    )
    trainer_pluribus = PokerTrainer(config_pluribus)
    
    results_pluribus = trainer_pluribus.train_step(game_results)
    print(f"   Unique info sets: {results_pluribus['unique_info_sets']:,}")
    print(f"   Info sets processed: {results_pluribus['info_sets_processed']:,}")
    
    # Comparison
    print("\n📊 Comparison Results:")
    print("=" * 40)
    compression_ratio = results_fine['unique_info_sets'] / max(results_pluribus['unique_info_sets'], 1)
    print(f"   Fine bucketing unique sets: {results_fine['unique_info_sets']:,}")
    print(f"   Pluribus bucketing unique sets: {results_pluribus['unique_info_sets']:,}")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Memory reduction: {(1 - 1/compression_ratio)*100:.1f}%")
    
    # Expected results
    print("\n🎯 Expected Results:")
    print("   Fine bucketing: ~2,000-5,000 unique sets")
    print("   Pluribus bucketing: ~100-400 unique sets")
    print("   Compression: 10-50x reduction")
    
    return results_fine, results_pluribus

def test_bucket_distribution():
    """Test bucket distribution and uniqueness"""
    print("\n🔍 Testing Bucket Distribution")
    print("=" * 40)
    
    from poker_bot.core.pluribus_bucket_gpu import pluribus_bucket_kernel_wrapper, estimate_unique_buckets
    
    # Generate test data
    batch_size = 10000
    hole_cards = cp.random.randint(0, 52, (batch_size, 2))
    community_cards = cp.random.randint(-1, 52, (batch_size, 5))
    positions = cp.random.randint(0, 6, (batch_size,))
    pot_sizes = cp.random.uniform(10, 100, (batch_size,))
    stack_sizes = cp.random.uniform(50, 200, (batch_size,))
    num_actives = cp.random.randint(2, 7, (batch_size,))
    
    # Generate buckets
    bucket_ids = pluribus_bucket_kernel_wrapper(
        hole_cards, community_cards, positions,
        pot_sizes, stack_sizes, num_actives
    )
    
    # Analyze distribution
    unique_buckets = len(cp.unique(bucket_ids))
    
    # Check if bucket IDs are in manageable range
    max_bucket_id = cp.max(bucket_ids)
    if max_bucket_id > 10000:
        print(f"   ⚠️  Warning: Bucket IDs too high: {max_bucket_id}")
        print(f"   Limiting analysis to first 10000 buckets")
        bucket_ids = cp.clip(bucket_ids, 0, 9999)
    
    # Use histogram instead of bincount to avoid memory issues
    bucket_counts, _ = cp.histogram(bucket_ids.flatten(), bins=cp.arange(10001))
    max_count = cp.max(bucket_counts)
    min_count = cp.min(bucket_counts[bucket_counts > 0])  # Ignore empty buckets
    avg_count = cp.mean(bucket_counts[bucket_counts > 0])  # Only non-empty buckets
    
    print(f"   Total hands: {batch_size:,}")
    print(f"   Unique buckets: {unique_buckets:,}")
    print(f"   Estimated max buckets: {estimate_unique_buckets():,}")
    print(f"   Bucket utilization: {unique_buckets/estimate_unique_buckets()*100:.1f}%")
    print(f"   Max hands per bucket: {int(max_count)}")
    print(f"   Min hands per bucket: {int(min_count)}")
    print(f"   Avg hands per bucket: {avg_count:.1f}")
    
    # Show some example buckets
    print(f"\n   Example bucket IDs: {bucket_ids[:10].tolist()}")

if __name__ == "__main__":
    try:
        # Test bucketing comparison
        results_fine, results_pluribus = test_pluribus_bucketing()
        
        # Test bucket distribution
        test_bucket_distribution()
        
        print("\n✅ Pluribus bucketing test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 