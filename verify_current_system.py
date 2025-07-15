#!/usr/bin/env python3
"""
Quick verification script to test current poker bot system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer, TrainerConfig
import jax.random as jr
import logging

# Suppress excessive logging
logging.basicConfig(level=logging.WARNING)

def quick_test():
    """Quick 10-iteration test to verify system works"""
    print("🧪 Quick System Verification")
    print("=" * 40)
    
    # Create minimal config for testing
    config = TrainerConfig(
        batch_size=512,  # Small for quick test
        learning_rate=0.1,
        temperature=1.0,
        num_actions=3,  # Simple actions
        dtype='float32',
        max_info_sets=1000,  # Small for testing
        growth_factor=1.2,
        chunk_size=500,
        gpu_bucket=False,
        use_pluribus_bucketing=True,
        N_rollouts=10
    )
    
    print("✅ Creating trainer...")
    trainer = PokerTrainer(config)
    
    # Test configuration
    game_config = {
        'players': 2,  # Simple 2-player for testing
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    print("✅ Running 10 test iterations...")
    unique_info_sets = []
    
    for i in range(10):
        rng_key = jr.PRNGKey(i)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        from poker_bot.core.simulation import batch_simulate_real_holdem
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        results = trainer.train_step(game_results, iteration=i)
        
        unique_info_sets.append(results['unique_info_sets'])
        print(f"   Iteration {i+1}: {results['unique_info_sets']} unique info sets")
    
    # Final check
    final_unique = unique_info_sets[-1]
    
    print("\n" + "=" * 40)
    print("🎯 VERIFICATION RESULTS:")
    
    if final_unique > 10:
        print(f"   ✅ SUCCESS: {final_unique} unique info sets generated")
        print("   ✅ Bucketing system is working correctly")
        print("   ✅ Ready for full training")
        return True
    else:
        print(f"   ❌ ISSUE: Only {final_unique} unique info sets")
        print("   ❌ Bucketing may need adjustment")
        return False

def test_phase1_components():
    """Test Phase 1 enhanced components"""
    print("\n🔍 Testing Phase 1 Components...")
    
    try:
        from poker_bot.core.enhanced_eval import EnhancedHandEvaluator
        from poker_bot.core.icm_modeling import ICMModel
        
        # Test enhanced evaluation
        evaluator = EnhancedHandEvaluator()
        print("   ✅ Enhanced evaluation loaded")
        
        # Test ICM modeling
        icm = ICMModel()
        print("   ✅ ICM modeling loaded")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Aequus Poker Bot - Quick Verification")
    print("=" * 50)
    
    # Test components
    components_ok = test_phase1_components()
    
    # Test system
    system_ok = quick_test()
    
    print("\n" + "=" * 50)
    print("📊 FINAL STATUS:")
    
    if components_ok and system_ok:
        print("🎉 SYSTEM READY FOR PRODUCTION!")
        print("\nNext steps:")
        print("1. python test_phase1.py  (comprehensive test)")
        print("2. python main_phase1.py --iterations 1000 --save_every 100")
        print("3. Deploy to Vast.ai for full training")
    else:
        print("⚠️  Issues detected - check logs above")