#!/usr/bin/env python3
"""
Super Bot Quick Test Script
Test all phases of the super-intelligent poker bot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cupy as cp
import numpy as np
import time

def test_all_phases():
    """Test all phases of the super bot"""
    print("🧪 Testing Super-Intelligent Bot - All Phases")
    print("=" * 50)
    
    try:
        # Test Phase 1: Enhanced Evaluation
        print("🎯 Phase 1: Enhanced Hand Evaluation...")
        from poker_bot.core.enhanced_eval import EnhancedHandEvaluator
        
        evaluator = EnhancedHandEvaluator()
        hole_cards = cp.random.randint(0, 52, (1000, 2))
        community_cards = cp.random.randint(-1, 52, (1000, 5))
        
        start = time.time()
        strengths = evaluator.enhanced_hand_strength(hole_cards, community_cards)
        elapsed = time.time() - start
        
        print(f"   ✅ Enhanced evaluation: {1000/elapsed:.0f} evals/sec")
        print(f"   ✅ Strength range: {cp.min(strengths)}-{cp.max(strengths)}")
        
        # Test Phase 2: History-Aware Bucketing
        print("🧠 Phase 2: History-Aware Bucketing...")
        from poker_bot.core.history_aware_bucketing import HistoryAwareBucketing
        
        bucketing = HistoryAwareBucketing()
        buckets = bucketing.create_history_buckets(
            hole_cards, community_cards,
            cp.random.randint(0, 6, 1000),
            cp.random.uniform(10, 100, 1000),
            cp.random.uniform(5, 50, 1000),
            cp.random.randint(2, 7, 1000),
            ['BET_50', 'CALL']
        )
        
        unique_buckets = len(cp.unique(buckets))
        print(f"   ✅ History buckets: {unique_buckets} unique")
        print(f"   ✅ Compression: {1000/unique_buckets:.1f}x")
        
        # Test Phase 3: Advanced MCCFR
        print("🎯 Phase 3: Advanced MCCFR...")
        from poker_bot.core.advanced_mccfr import AdvancedMCCFR
        
        mccfr = AdvancedMCCFR()
        keys = cp.random.randint(0, 2**32, 1000, dtype=cp.uint64)
        cf_values = mccfr.external_sampling_mccfr(keys, N_rollouts=10)
        
        print(f"   ✅ MCCFR: {cf_values.shape} cf_values generated")
        
        # Test Phase 4: Production Optimization
        print("🚀 Phase 4: Production Optimization...")
        from poker_bot.core.production_optimization import ProductionBucketing
        
        prod = ProductionBucketing()
        prod_buckets = prod.create_production_buckets(
            hole_cards, community_cards,
            cp.random.randint(0, 6, 1000),
            cp.random.uniform(10, 100, 1000),
            cp.random.uniform(5, 50, 1000),
            cp.random.randint(2, 7, 1000),
            ['BET_50', 'CALL'],
            cp.random.randint(0, 10, 1000),
            cp.random.random(1000)
        )
        
        prod_unique = len(cp.unique(prod_buckets))
        print(f"   ✅ Production buckets: {prod_unique} unique")
        print(f"   ✅ 4M bucket system: {prod_unique} used")
        
        # Test ICM Modeling
        print("💰 ICM Modeling...")
        from poker_bot.core.icm_modeling import ICMModel
        
        icm = ICMModel()
        adjustments = icm.get_icm_adjustment(
            cp.random.uniform(5, 200, 1000),
            cp.random.randint(0, 6, 1000),
            cp.random.uniform(10, 100, 1000)
        )
        
        print(f"   ✅ ICM: {1000/(time.time()-start):.0f} calcs/sec")
        print(f"   ✅ Adjustment range: {cp.min(adjustments):.3f}-{cp.max(adjustments):.3f}")
        
        print("\n🎉 All phases tested successfully!")
        print("🚀 Super bot is ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_phases()
    if success:
        print("\n✅ Super bot ready for deployment!")
        print("   Run: python main_super_bot.py --iterations 100")
        print("   Or:  bash scripts/deploy_super_bot.sh")
    else:
        print("\n❌ Please fix the errors above")
        sys.exit(1)