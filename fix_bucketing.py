#!/usr/bin/env python3
"""
🚨 URGENT FIX - Bucketing Issue Resolution
Fixes the bucketing problem causing all states to map to same bucket
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer, TrainerConfig
import pickle
import numpy as np

def create_fixed_config():
    """Create configuration that fixes bucketing issues"""
    return TrainerConfig(
        batch_size=8192,  # Smaller for RTX 3090
        learning_rate=0.05,
        temperature=1.0,
        num_actions=14,
        dtype='float32',  # Use float32 instead of bfloat16
        max_info_sets=50000,  # Start smaller for debugging
        growth_factor=1.5,
        chunk_size=10000,
        gpu_bucket=True,  # Enable proper GPU bucketing
        use_pluribus_bucketing=False,  # Use enhanced bucketing
        N_rollouts=100  # Reduced for stability
    )

def test_bucketing():
    """Test bucketing system"""
    print("🧪 Testing Bucketing System")
    print("=" * 40)
    
    config = create_fixed_config()
    trainer = PokerTrainer(config)
    
    # Test with small batch
    import jax.random as jr
    rng_key = jr.PRNGKey(42)
    rng_keys = jr.split(rng_key, 100)
    
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    from poker_bot.core.simulation import batch_simulate_real_holdem
    game_results = batch_simulate_real_holdem(rng_keys, game_config)
    
    # Test bucketing
    results = trainer.train_step(game_results, iteration=0)
    
    print(f"✅ Unique info sets: {results['unique_info_sets']}")
    print(f"✅ Info sets processed: {results['info_sets_processed']}")
    print(f"✅ Array size: {results['array_size']}")
    
    if results['unique_info_sets'] > 1:
        print("🎉 Bucketing working correctly!")
        return True
    else:
        print("❌ Bucketing still failing")
        return False

def fix_existing_models():
    """Fix existing model configuration"""
    print("🔧 Fixing existing models...")
    
    # Find latest model
    models = glob.glob("models/3090_super_bot_*.pkl")
    if not models:
        print("❌ No models found")
        return
    
    latest = max(models, key=os.path.getctime)
    print(f"📂 Fixing: {latest}")
    
    try:
        with open(latest, 'rb') as f:
            model = pickle.load(f)
        
        # Update configuration
        model['config'] = create_fixed_config()
        
        # Save fixed version
        fixed_path = latest.replace('.pkl', '_fixed.pkl')
        with open(fixed_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Fixed model saved: {fixed_path}")
        
    except Exception as e:
        print(f"❌ Error fixing model: {e}")

if __name__ == "__main__":
    print("🚨 URGENT BUCKETING FIX")
    print("=" * 40)
    
    # Test new configuration
    success = test_bucketing()
    
    if success:
        print("\n✅ Configuration fixed!")
        print("🚀 Ready to restart training with:")
        print("  python main_super_bot.py --iterations 10000 --save_every 1000")
    else:
        print("\n❌ Need further debugging")
        
    # Also fix existing models
    import glob
    fix_existing_models()