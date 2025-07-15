#!/usr/bin/env python3
"""
⚡ Quick Model Comparison Tool
Fast comparison for ongoing training
"""

import sys
import os
import pickle
import glob
from pathlib import Path

def quick_compare():
    """Quick comparison of latest models"""
    
    # Find all super bot models
    model_pattern = "models/3090_super_bot*.pkl"
    models = glob.glob(model_pattern)
    
    if not models:
        print("❌ No models found")
        return
    
    # Sort by iteration
    models.sort(key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))) if any(c.isdigit() for c in Path(x).stem) else 0)
    
    print("🚀 Quick Model Comparison")
    print("=" * 40)
    
    # Load latest 3 models
    latest_models = models[-3:] if len(models) >= 3 else models
    
    for model_path in latest_models:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            name = Path(model_path).stem
            iteration = model.get('iteration', 0)
            info_sets = model.get('unique_info_sets', 0)
            games = model.get('total_games', 0)
            file_size = os.path.getsize(model_path) / (1024*1024)
            
            print(f"\n{name}:")
            print(f"  Iteration: {iteration:,}")
            print(f"  Info sets: {info_sets:,}")
            print(f"  Games: {games:,}")
            print(f"  Size: {file_size:.1f} MB")
            
            # Quick quality check
            if 'q_values' in model and model['q_values'].size > 0:
                q_vals = model['q_values']
                non_zero = q_vals[q_vals != 0]
                if len(non_zero) > 0:
                    avg_q = float(non_zero.mean())
                    print(f"  Avg Q-value: {avg_q:.3f}")
            
        except Exception as e:
            print(f"❌ Error with {model_path}: {e}")
    
    # Show improvement
    if len(latest_models) >= 2:
        try:
            with open(latest_models[-1], 'rb') as f:
                latest = pickle.load(f)
            with open(latest_models[0], 'rb') as f:
                first = pickle.load(f)
            
            latest_info = latest.get('unique_info_sets', 0)
            first_info = first.get('unique_info_sets', 0)
            improvement = latest_info - first_info
            
            print(f"\n📈 Improvement:")
            print(f"  Info sets growth: {improvement:,}")
            print(f"  Latest vs first: {latest_info:,} vs {first_info:,}")
            
        except:
            pass
    
    print("\n✅ Ready for next comparison!")

if __name__ == "__main__":
    quick_compare()