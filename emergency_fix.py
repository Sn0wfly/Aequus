#!/usr/bin/env python3
"""
🚨 EMERGENCY FIX - Immediate Solution
Quick fix for bucketing and training issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_emergency_config():
    """Create emergency configuration"""
    print("🚨 Creating emergency configuration...")
    
    # Create new main with fixed parameters
    emergency_main = '''
#!/usr/bin/env python3
"""
Emergency Main - Fixed Configuration
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.core.simulation import batch_simulate_real_holdem
import jax.random as jr
import time
import logging

logging.basicConfig(level=logging.INFO)

def create_working_config():
    """Working configuration for RTX 3090"""
    return TrainerConfig(
        batch_size=1024,  # Much smaller for debugging
        learning_rate=0.1,
        temperature=1.0,
        num_actions=3,  # Simplified actions
        dtype='float32',
        max_info_sets=1000,  # Start small
        growth_factor=1.2,
        chunk_size=1000,
        gpu_bucket=False,
        use_pluribus_bucketing=True,  # Use working bucketing
        N_rollouts=10
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="models/emergency_bot")
    
    args = parser.parse_args()
    
    config = create_working_config()
    trainer = PokerTrainer(config)
    
    game_config = {
        'players': 2,  # Simplified for testing
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    for iteration in range(args.iterations):
        rng_key = jr.PRNGKey(iteration)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        results = trainer.train_step(game_results, iteration=iteration)
        
        print(f"Iteration {iteration+1}: {results['unique_info_sets']} unique info sets")
        
        if (iteration + 1) % args.save_every == 0:
            trainer.save_model(f"{args.save_path}_checkpoint_{iteration+1}.pkl")
    
    trainer.save_model(f"{args.save_path}_final.pkl")

if __name__ == "__main__":
    main()
'''
    
    with open('emergency_main.py', 'w') as f:
        f.write(emergency_main)
    
    print("✅ Emergency main created: emergency_main.py")
    print("🚀 Run: python emergency_main.py --iterations 100")

def create_debug_config():
    """Create debug configuration file"""
    debug_config = '''
# Debug configuration
num_iterations: 100
batch_size: 1024
learning_rate: 0.1
num_players: 2
starting_stack: 100.0
small_blind: 1.0
big_blind: 2.0
num_card_buckets: 1000
bet_sizes: [0.5, 1.0, 2.0]
max_memory_gb: 2
'''
    
    with open('config/debug_config.yaml', 'w') as f:
        f.write(debug_config)
    
    print("✅ Debug config created: config/debug_config.yaml")

def main():
    print("🚨 EMERGENCY FIX IMPLEMENTED")
    print("=" * 40)
    
    create_emergency_config()
    create_debug_config()
    
    print("\n📋 Next Steps:")
    print("1. Stop current training")
    print("2. Run: python emergency_main.py --iterations 100")
    print("3. Verify unique info sets > 10")
    print("4. Scale up gradually")
    
    print("\n🔧 Commands:")
    print("  python emergency_main.py --iterations 100 --save_every 20")
    print("  python compare_models.py models/emergency_bot_*.pkl")

if __name__ == "__main__":
    main()