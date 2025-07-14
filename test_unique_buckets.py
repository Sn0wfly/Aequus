#!/usr/bin/env python3
"""
Test script for unique bucket generation.
Uses correct data generation with unique subkeys per batch element.
"""

import jax
import jax.numpy as jnp
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

def test_unique_buckets():
    """Test that we generate unique buckets with proper data generation."""
    print("🧪 Testing unique bucket generation...")
    
    trainer = PokerTrainer(TrainerConfig(batch_size=8192))
    
    for i in range(125):
        # ✅ CORRECT: Use unique subkey per batch element
        base_rng = jax.random.PRNGKey(i)
        rngs = jax.random.split(base_rng, 8192)  # 8192 subclaves únicas
        
        # Generate unique data per batch element
        game = {
            'hole_cards': jax.vmap(lambda r: jax.random.randint(r, (6,2), 0, 52))(rngs),
            'final_community': jax.vmap(lambda r: jax.random.randint(r, (5,), 0, 52))(rngs),
            'payoffs': jax.vmap(lambda r: jax.random.normal(r, (6,)))(rngs),
            'final_pot': jax.vmap(lambda r: jax.random.uniform(r, ()))(rngs)
        }
        
        trainer.train_step(game)
        
        # Print progress every 25 iterations
        if (i + 1) % 25 == 0:
            print(f"   Iteration {i+1}/125: {trainer.total_unique_info_sets:,} unique info-sets")
    
    print(f"\n🎯 Final result: {trainer.total_unique_info_sets:,} unique info-sets")
    print(f"   Expected: >1000 unique info-sets")
    print(f"   Success: {'✅' if trainer.total_unique_info_sets > 1000 else '❌'}")

if __name__ == "__main__":
    test_unique_buckets() 