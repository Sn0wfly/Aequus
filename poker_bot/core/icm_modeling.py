"""
🎯 ICM (Independent Chip Model) Modeling for Tournament Poker
Phase 1 Enhancement: Pre-computed ICM tables for push/fold scenarios
"""

import cupy as cp
import numpy as np
from typing import Tuple, Dict
import pickle

class ICMModel:
    """
    Fast ICM calculations for tournament scenarios
    Optimized for GPU deployment with pre-computed tables
    """
    
    def __init__(self):
        self.icm_tables = self._precompute_icm_tables()
        self.stack_classes = self._define_stack_classes()
        
    def _define_stack_classes(self) -> Dict[int, Tuple[float, float]]:
        """Define stack depth classes for ICM modeling"""
        return {
            0: (0.0, 5.0),      # Critical short stack
            1: (5.0, 10.0),     # Short stack
            2: (10.0, 15.0),    # Medium-short
            3: (15.0, 25.0),    # Medium
            4: (25.0, 40.0),    # Medium-deep
            5: (40.0, 60.0),    # Deep
            6: (60.0, 80.0),    # Very deep
            7: (80.0, 100.0),   # Super deep
            8: (100.0, 150.0),  # Monster stack
            9: (150.0, 1000.0)  # Chip leader
        }
    
    def _precompute_icm_tables(self) -> Dict[str, cp.ndarray]:
        """Pre-compute ICM tables for fast lookup"""
        tables = {}
        
        # Pre-compute push/fold equity for common scenarios
        # Key: (stack_class, position, num_players, pot_size)
        
        # 6-max tournament scenarios
        for num_players in [2, 3, 4, 5, 6]:
            table_name = f"icm_6max_{num_players}"
            tables[table_name] = self._compute_icm_table(num_players)
            
        return tables
    
    def _compute_icm_table(self, num_players: int) -> cp.ndarray:
        """Compute ICM table for given player count"""
        # Simplified ICM calculation
        # Returns equity multiplier for push/fold decisions
        
        table_size = 10 * 6 * 20  # 10 stack classes × 6 positions × 20 pot sizes
        table = cp.zeros((10, 6, 20), dtype=cp.float32)
        
        for stack_class in range(10):
            for position in range(6):
                for pot_idx in range(20):
                    # Simplified ICM calculation
                    base_equity = 0.5
                    
                    # Stack depth adjustment
                    stack_factor = (10 - stack_class) / 10.0
                    
                    # Position adjustment
                    pos_factor = (6 - position) / 6.0
                    
                    # Pot size adjustment
                    pot_factor = pot_idx / 20.0
                    
                    # Combine factors
                    icm_equity = base_equity + (stack_factor * pos_factor * pot_factor * 0.3)
                    table[stack_class, position, pot_idx] = cp.clip(icm_equity, 0.1, 0.9)
                    
        return table
    
    def get_icm_adjustment(self, stack_sizes: cp.ndarray, 
                          positions: cp.ndarray, 
                          pot_sizes: cp.ndarray,
                          num_players: int = 6) -> cp.ndarray:
        """
        Get ICM adjustment for push/fold decisions
        Args:
            stack_sizes: (batch,) stack sizes in BB
            positions: (batch,) player positions (0-5)
            pot_sizes: (batch,) current pot sizes in BB
            num_players: number of active players
        Returns:
            icm_adjustment: (batch,) adjustment factors
        """
        batch_size = stack_sizes.shape[0]
        
        # Convert stack sizes to classes
        stack_classes = cp.zeros(batch_size, dtype=cp.int32)
        for class_idx, (min_stack, max_stack) in self.stack_classes.items():
            mask = (stack_sizes >= min_stack) & (stack_sizes < max_stack)
            stack_classes = cp.where(mask, class_idx, stack_classes)
        
        # Clamp positions
        positions = cp.clip(positions, 0, 5)
        
        # Convert pot sizes to indices
        pot_indices = cp.clip((pot_sizes / 10.0).astype(cp.int32), 0, 19)
        
        # Get ICM adjustments
        table_name = f"icm_6max_{num_players}"
        if table_name in self.icm_tables:
            table = self.icm_tables[table_name]
            adjustments = table[stack_classes, positions, pot_indices]
        else:
            adjustments = cp.full(batch_size, 0.5, dtype=cp.float32)
            
        return adjustments
    
    def calculate_push_fold_equity(self, hole_cards: cp.ndarray,
                                 stack_sizes: cp.ndarray,
                                 positions: cp.ndarray,
                                 pot_sizes: cp.ndarray) -> cp.ndarray:
        """
        Calculate push/fold equity with ICM adjustment
        Args:
            hole_cards: (batch, 2) hole cards
            stack_sizes: (batch,) stack sizes in BB
            positions: (batch,) player positions
            pot_sizes: (batch,) pot sizes in BB
        Returns:
            equity: (batch,) push/fold equity [0,1]
        """
        # Base hand strength
        from .enhanced_eval import EnhancedHandEvaluator
        evaluator = EnhancedHandEvaluator()
        base_strength = evaluator.enhanced_hand_strength(hole_cards, 
                                                        cp.full((hole_cards.shape[0], 5), -1))
        
        # Normalize to [0,1]
        base_equity = base_strength / 1000.0
        
        # ICM adjustment
        icm_adjustment = self.get_icm_adjustment(stack_sizes, positions, pot_sizes)
        
        # Combine base equity with ICM adjustment
        final_equity = base_equity * icm_adjustment
        
        return cp.clip(final_equity, 0.0, 1.0)

# ICM-aware bucketing for Phase 1
class ICMAwareBucketing:
    """
    Enhanced bucketing with ICM considerations
    """
    
    def __init__(self):
        self.icm_model = ICMModel()
        
    def create_icm_buckets(self, hole_cards: cp.ndarray,
                          stack_sizes: cp.ndarray,
                          positions: cp.ndarray,
                          pot_sizes: cp.ndarray,
                          num_actives: cp.ndarray) -> cp.ndarray:
        """
        Create ICM-aware buckets
        Args:
            hole_cards: (batch, 2) hole cards
            stack_sizes: (batch,) stack sizes
            positions: (batch,) positions
            pot_sizes: (batch,) pot sizes
            num_actives: (batch,) active players
        Returns:
            bucket_ids: (batch,) enhanced bucket IDs
        """
        batch_size = hole_cards.shape[0]
        
        # Base bucket calculation
        from .pluribus_bucket_gpu import pluribus_bucket_kernel_wrapper
        base_buckets = pluribus_bucket_kernel_wrapper(
            hole_cards, 
            cp.full((batch_size, 5), -1),  # No community cards for preflop
            positions,
            pot_sizes,
            stack_sizes,
            num_actives
        )
        
        # ICM adjustment
        icm_equity = self.icm_model.calculate_push_fold_equity(
            hole_cards, stack_sizes, positions, pot_sizes
        )
        
        # Create enhanced buckets
        icm_class = (icm_equity * 10).astype(cp.uint32)  # 10 ICM classes
        enhanced_buckets = base_buckets * 10 + icm_class
        
        return enhanced_buckets % 50000  # Keep in Phase 1 range

if __name__ == "__main__":
    # Test ICM modeling
    icm = ICMModel()
    
    # Test data
    batch_size = 1000
    stack_sizes = cp.random.uniform(5, 200, batch_size)
    positions = cp.random.randint(0, 6, batch_size)
    pot_sizes = cp.random.uniform(10, 100, batch_size)
    
    # Benchmark
    start = time.time()
    adjustments = icm.get_icm_adjustment(stack_sizes, positions, pot_sizes)
    elapsed = time.time() - start
    
    print(f"ICM calculation: {batch_size/elapsed:.0f} calculations/sec")
    print(f"Adjustment range: {cp.min(adjustments):.3f}-{cp.max(adjustments):.3f}")