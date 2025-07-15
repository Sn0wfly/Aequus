#!/usr/bin/env python3
"""
Main training script for Aequus - Super-Pluribus GPU Poker AI
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_super_pluribus_config():
    """Create super-Pluribus configuration for H100 training"""
    return TrainerConfig(
        batch_size=32768,        # H100 optimized - fills VRAM
        learning_rate=0.05,      # Pluribus-like learning rate
        temperature=1.0,
        num_actions=14,          # Pluribus-like action set
        dtype='bfloat16',
        accumulation_dtype='float32',
        max_info_sets=25000,     # 25k buckets fixed for super-human quality
        growth_factor=1.5,
        chunk_size=20000,
        gpu_bucket=False,
        use_pluribus_bucketing=True,  # Super-Pluribus bucketing
        N_rollouts=500           # GPU-accelerated rollouts per bucket
    )

def create_debug_config():
    """Create debug configuration with reduced batch size"""
    return TrainerConfig(
        batch_size=512,          # Debug - small batch size
        learning_rate=0.05,      # Pluribus-like learning rate
        temperature=1.0,
        num_actions=14,          # Pluribus-like action set
        dtype='bfloat16',
        accumulation_dtype='float32',
        max_info_sets=25000,     # 25k buckets fixed for super-human quality
        growth_factor=1.5,
        chunk_size=20000,
        gpu_bucket=False,
        use_pluribus_bucketing=True,  # Super-Pluribus bucketing
        N_rollouts=100           # Debug - reduced rollouts
    )

def create_fine_bucketing_config():
    """Create fine bucketing configuration for comparison"""
    return TrainerConfig(
        batch_size=8192,
        learning_rate=0.1,
        temperature=1.0,
        num_actions=4,
        dtype='bfloat16',
        accumulation_dtype='float32',
        max_info_sets=1000000,
        growth_factor=1.5,
        chunk_size=20000,
        gpu_bucket=False,
        use_pluribus_bucketing=False,  # Fine bucketing
        N_rollouts=100
    )

def train_model(config: TrainerConfig, num_iterations: int, save_every: int, save_path: str):
    """Train the poker model"""
    logger.info("🚀 Starting Aequus Super-Pluribus Training")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Batch size: {config.batch_size:,}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Num actions: {config.num_actions}")
    logger.info(f"  Max info sets: {config.max_info_sets:,}")
    logger.info(f"  N rollouts: {config.N_rollouts}")
    logger.info(f"  Bucketing: {'Super-Pluribus' if config.use_pluribus_bucketing else 'Fine'}")
    logger.info(f"  Iterations: {num_iterations:,}")
    logger.info(f"  Save every: {save_every}")
    logger.info(f"  Save path: {save_path}")
    
    # Initialize trainer
    trainer = PokerTrainer(config)
    
    # Game configuration
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    # Training loop
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Generate random keys for this iteration
        rng_key = jr.PRNGKey(iteration)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        # Simulate games
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        
        # Training step
        results = trainer.train_step(game_results)
        
        # Log progress
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (iteration + 1) / elapsed
            eta = (num_iterations - iteration - 1) / rate if rate > 0 else 0
            
            logger.info(f"Iteration {iteration+1:,}/{num_iterations:,}")
            logger.info(f"  Unique info sets: {results['unique_info_sets']:,}")
            logger.info(f"  Info sets processed: {results['info_sets_processed']:,}")
            logger.info(f"  Avg payoff: {results['avg_payoff']:.3f}")
            logger.info(f"  Strategy entropy: {float(results['strategy_entropy']):.3f}")
            logger.info(f"  Elapsed: {elapsed:.1f}s, Rate: {rate:.1f} it/s, ETA: {eta:.1f}s")
            logger.info("-" * 40)
        
        # Save checkpoint
        if (iteration + 1) % save_every == 0:
            checkpoint_path = f"{save_path}_checkpoint_{iteration+1}.pkl"
            trainer.save_model(checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = f"{save_path}_final.pkl"
    trainer.save_model(final_path)
    
    total_time = time.time() - start_time
    logger.info("🎉 Training completed!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    logger.info(f"Final model saved: {final_path}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Aequus Super-Pluribus Training")
    parser.add_argument("--config", choices=["super_pluribus", "fine", "debug"], 
                       default="super_pluribus", help="Training configuration")
    parser.add_argument("--iterations", type=int, default=1000, 
                       help="Number of training iterations")
    parser.add_argument("--save_every", type=int, default=100, 
                       help="Save checkpoint every N iterations")
    parser.add_argument("--save_path", type=str, default="aequus_model", 
                       help="Base path for saving models")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config == "super_pluribus":
        config = create_super_pluribus_config()
    elif args.config == "debug":
        config = create_debug_config()
    else:
        config = create_fine_bucketing_config()
    
    # Train model
    trainer = train_model(config, args.iterations, args.save_every, args.save_path)
    
    return trainer

if __name__ == "__main__":
    main() 