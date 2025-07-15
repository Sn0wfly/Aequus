#!/usr/bin/env python3
"""
🔍 Model Comparison Tool for Super Bot
Compare PKL models to track improvement and quality
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

class ModelComparator:
    """Compare poker bot models to track improvement"""
    
    def __init__(self):
        self.metrics = {}
        
    def load_model(self, model_path: str) -> Dict:
        """Load a model file"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            return None
    
    def compare_models(self, model_paths: List[str]) -> Dict:
        """Compare multiple models"""
        results = {}
        
        for path in model_paths:
            model = self.load_model(path)
            if model is None:
                continue
                
            model_name = Path(path).stem
            
            results[model_name] = {
                'iteration': model.get('iteration', 0),
                'unique_info_sets': model.get('unique_info_sets', 0),
                'total_games': model.get('total_games', 0),
                'q_values_shape': model['q_values'].shape if 'q_values' in model else (0, 0),
                'strategies_shape': model['strategies'].shape if 'strategies' in model else (0, 0),
                'file_size_mb': os.path.getsize(path) / (1024 * 1024),
                'avg_payoff': self._calculate_avg_payoff(model),
                'strategy_entropy': self._calculate_entropy(model),
                'convergence_rate': self._calculate_convergence(model),
                'action_distribution': self._analyze_actions(model)
            }
        
        return results
    
    def _calculate_avg_payoff(self, model: Dict) -> float:
        """Calculate average payoff from Q-values"""
        if 'q_values' not in model:
            return 0.0
        
        q_values = model['q_values']
        if q_values.size == 0:
            return 0.0
            
        # Use only non-zero Q-values
        non_zero = q_values[q_values != 0]
        if len(non_zero) == 0:
            return 0.0
            
        return float(np.mean(non_zero))
    
    def _calculate_entropy(self, model: Dict) -> float:
        """Calculate average strategy entropy"""
        if 'strategies' not in model:
            return 0.0
        
        strategies = model['strategies']
        if strategies.size == 0:
            return 0.0
            
        # Calculate entropy for each strategy
        epsilon = 1e-8
        entropy = -np.sum(strategies * np.log(strategies + epsilon), axis=1)
        
        # Return average entropy
        return float(np.mean(entropy))
    
    def _calculate_convergence(self, model: Dict) -> float:
        """Calculate convergence metric"""
        if 'strategies' not in model:
            return 0.0
        
        strategies = model['strategies']
        if strategies.size == 0:
            return 0.0
            
        # Measure how concentrated strategies are
        max_probs = np.max(strategies, axis=1)
        return float(np.mean(max_probs))
    
    def _analyze_actions(self, model: Dict) -> Dict[str, float]:
        """Analyze action distribution"""
        if 'strategies' not in model:
            return {}
        
        strategies = model['strategies']
        if strategies.size == 0:
            return {}
            
        # Get most probable action for each state
        actions = np.argmax(strategies, axis=1)
        
        # Count action frequencies
        unique, counts = np.unique(actions, return_counts=True)
        total = len(actions)
        
        action_names = ['FOLD', 'CHECK', 'CALL', 'BET_33', 'BET_50', 'BET_75', 
                       'BET_100', 'BET_150', 'BET_200', 'RAISE', 'ALL_IN', 'LIMP', 
                       '3BET', '4BET']
        
        distribution = {}
        for action, count in zip(unique, counts):
            if action < len(action_names):
                distribution[action_names[action]] = count / total
        
        return distribution
    
    def generate_report(self, results: Dict) -> str:
        """Generate comparison report"""
        if not results:
            return "No models to compare"
        
        report = []
        report.append("🎯 Super Bot Model Comparison Report")
        report.append("=" * 50)
        
        # Sort by iteration
        sorted_models = sorted(results.items(), key=lambda x: x[1]['iteration'])
        
        # Performance metrics
        report.append("\n📊 Performance Metrics:")
        report.append("-" * 30)
        
        for name, data in sorted_models:
            report.append(f"\n{name}:")
            report.append(f"  Iteration: {data['iteration']:,}")
            report.append(f"  Unique info sets: {data['unique_info_sets']:,}")
            report.append(f"  Total games: {data['total_games']:,}")
            report.append(f"  File size: {data['file_size_mb']:.1f} MB")
            report.append(f"  Avg payoff: {data['avg_payoff']:.3f}")
            report.append(f"  Strategy entropy: {data['strategy_entropy']:.3f}")
            report.append(f"  Convergence: {data['convergence_rate']:.3f}")
        
        # Improvement tracking
        if len(sorted_models) > 1:
            report.append("\n📈 Improvement Tracking:")
            report.append("-" * 30)
            
            first = sorted_models[0][1]
            last = sorted_models[-1][1]
            
            improvement = {
                'info_sets': last['unique_info_sets'] - first['unique_info_sets'],
                'payoff': last['avg_payoff'] - first['avg_payoff'],
                'entropy': last['strategy_entropy'] - first['strategy_entropy'],
                'convergence': last['convergence_rate'] - first['convergence_rate']
            }
            
            report.append(f"Info sets growth: {improvement['info_sets']:,}")
            report.append(f"Payoff improvement: {improvement['payoff']:.3f}")
            report.append(f"Entropy change: {improvement['entropy']:.3f}")
            report.append(f"Convergence improvement: {improvement['convergence']:.3f}")
        
        # Action distribution
        if len(sorted_models) >= 1:
            latest = sorted_models[-1][1]
            if latest['action_distribution']:
                report.append("\n🎲 Latest Action Distribution:")
                report.append("-" * 30)
                for action, prob in sorted(latest['action_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    report.append(f"  {action}: {prob:.1%}")
        
        return "\n".join(report)
    
    def plot_progress(self, results: Dict, output_path: str = None):
        """Create progress visualization"""
        if len(results) < 2:
            print("Need at least 2 models for plotting")
            return
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['iteration'])
        iterations = [data['iteration'] for _, data in sorted_models]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Info sets growth
        info_sets = [data['unique_info_sets'] for _, data in sorted_models]
        ax1.plot(iterations, info_sets, 'b-o')
        ax1.set_title('Unique Info Sets Growth')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Unique Info Sets')
        ax1.grid(True)
        
        # Payoff improvement
        payoffs = [data['avg_payoff'] for _, data in sorted_models]
        ax2.plot(iterations, payoffs, 'g-o')
        ax2.set_title('Average Payoff Improvement')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Payoff')
        ax2.grid(True)
        
        # Strategy entropy
        entropies = [data['strategy_entropy'] for _, data in sorted_models]
        ax3.plot(iterations, entropies, 'r-o')
        ax3.set_title('Strategy Entropy')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Entropy')
        ax3.grid(True)
        
        # Convergence
        convergence = [data['convergence_rate'] for _, data in sorted_models]
        ax4.plot(iterations, convergence, 'm-o')
        ax4.set_title('Convergence Rate')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Max Probability')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {output_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare poker bot models")
    parser.add_argument("models", nargs="+", help="Model files to compare")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output", help="Output file for plot")
    
    args = parser.parse_args()
    
    comparator = ModelComparator()
    results = comparator.compare_models(args.models)
    
    if not results:
        print("❌ No valid models found")
        return
    
    report = comparator.generate_report(results)
    print(report)
    
    if args.plot:
        comparator.plot_progress(results, args.output)

if __name__ == "__main__":
    main()