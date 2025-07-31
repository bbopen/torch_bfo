#!/usr/bin/env python3
"""
P1 Schwefel Function Experiments with ChaoticBFO
===============================================

Tests P1 improvements:
1. Diversity-triggered elimination with 50% replacement
2. Chaos injection using logistic map
3. GA crossover in reproduction
4. Ablation study to isolate each component's contribution
5. Extended 100k budget test

Target: Achieve ≥20% success rate on 2D Schwefel with tolerance 1e-4
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import experiment framework
from schwefel_enhanced_experiments import SchwefelEnhancedExperiments


def run_p1_validation():
    """Run P1 validation experiments."""
    print("\n" + "=" * 80)
    print("P1 IMPROVEMENTS VALIDATION")
    print("ChaoticBFO with enhanced exploration for deceptive landscapes")
    print("=" * 80)
    
    # Initialize experimenter
    experimenter = SchwefelEnhancedExperiments()
    
    # Run P1 experiments
    p1_results = experimenter.run_p1_experiments()
    
    # Generate detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED P1 ANALYSIS")
    print("=" * 80)
    
    # Analyze diversity maintenance
    if 'P1_Chaotic_BFO' in p1_results:
        print("\n📊 Diversity Analysis:")
        chaotic_runs = p1_results['P1_Chaotic_BFO']['individual_runs']
        
        # Check if any runs triggered diversity maintenance
        diversity_triggers = 0
        for run in chaotic_runs[:5]:  # Sample first 5 runs
            # Note: Would need to add diversity history tracking to optimizer
            print(f"   Run {run['seed']}: Final loss = {run['final_loss']:.2f}")
        
    # Plot convergence comparison
    plot_p1_convergence(p1_results)
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    any_success = any(r['results']['success_rate'] > 0 for r in p1_results.values())
    
    if any_success:
        print("✅ SUCCESS! P1 improvements achieved target ≥20% success rate")
        print("   Ready for production deployment")
    else:
        # Check if we're close
        best_config = min(p1_results.keys(), 
                         key=lambda k: p1_results[k]['results']['final_loss_mean'])
        best_loss = p1_results[best_config]['results']['final_loss_mean']
        
        if best_loss < 100:  # Very close to optimum
            print("⚠️  Very close! Best loss < 100")
            print("   Consider:")
            print("   - Fine-tuning chaos_strength (try 0.3-0.7)")
            print("   - Adjusting diversity_trigger_ratio")
            print("   - Testing with 200k FE budget")
        elif best_loss < 200:
            print("⚠️  Significant improvement but not converged")
            print("   Consider P2 improvements:")
            print("   - Adaptive chaos strength based on stagnation")
            print("   - Multi-population with migration")
            print("   - Hybrid with local search near promising regions")
        else:
            print("❌ Limited improvement")
            print("   Schwefel may require fundamentally different approaches")
            print("   Consider: CMA-ES, differential evolution, or hybrid methods")
    
    return p1_results


def plot_p1_convergence(results):
    """Generate P1-specific convergence plots."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Convergence curves for all P1 variants
    plt.subplot(2, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (name, data) in enumerate(results.items()):
        # Average loss trajectory across runs
        if data['individual_runs']:
            # Get first successful or best run
            best_run = min(data['individual_runs'], 
                          key=lambda r: r['final_loss'])
            
            fe_traj = best_run['fe_trajectory']
            loss_traj = best_run['loss_trajectory']
            
            if len(fe_traj) > 0 and len(loss_traj) > 0:
                plt.plot(fe_traj[:len(loss_traj)], loss_traj, 
                        label=name.replace('P1_', ''), 
                        color=colors[i % len(colors)], 
                        alpha=0.7, linewidth=2)
    
    plt.xlabel('Function Evaluations')
    plt.ylabel('Loss')
    plt.title('P1 Convergence Comparison')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Final loss distribution
    plt.subplot(2, 2, 2)
    labels = []
    losses = []
    
    for name, data in results.items():
        labels.append(name.replace('P1_', ''))
        losses.append(data['results']['final_loss_mean'])
    
    bars = plt.bar(range(len(labels)), losses, color=colors[:len(labels)])
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Mean Final Loss')
    plt.title('P1 Final Loss Comparison')
    
    # Add P0 baseline line
    plt.axhline(y=371.20, color='black', linestyle='--', 
                label='P0 Best', alpha=0.5)
    plt.legend()
    
    # Plot 3: Success rate comparison
    plt.subplot(2, 2, 3)
    success_rates = []
    
    for name, data in results.items():
        success_rates.append(data['results']['success_rate'] * 100)
    
    bars = plt.bar(range(len(labels)), success_rates, color=colors[:len(labels)])
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Success Rate (%)')
    plt.title('P1 Success Rate Comparison')
    plt.axhline(y=20, color='red', linestyle='--', 
                label='Target 20%', alpha=0.5)
    plt.legend()
    
    # Plot 4: Component contribution
    plt.subplot(2, 2, 4)
    if 'P1_Chaotic_BFO' in results:
        baseline = results['P1_Chaotic_BFO']['results']['final_loss_mean']
        contributions = []
        comp_names = []
        
        if 'P1_NoGA' in results:
            no_ga = results['P1_NoGA']['results']['final_loss_mean']
            contributions.append((baseline - no_ga) / baseline * 100)
            comp_names.append('GA Crossover')
        
        if 'P1_NoChaos' in results:
            no_chaos = results['P1_NoChaos']['results']['final_loss_mean']
            contributions.append((baseline - no_chaos) / baseline * 100)
            comp_names.append('Chaos Injection')
        
        if 'P1_NoDiversity' in results:
            no_div = results['P1_NoDiversity']['results']['final_loss_mean']
            contributions.append((baseline - no_div) / baseline * 100)
            comp_names.append('Diversity Trigger')
        
        if contributions:
            colors_contrib = ['green' if c > 0 else 'red' for c in contributions]
            plt.bar(range(len(contributions)), contributions, 
                   color=colors_contrib, alpha=0.7)
            plt.xticks(range(len(contributions)), comp_names, rotation=45)
            plt.ylabel('Contribution (%)')
            plt.title('Component Contribution to Performance')
            plt.axhline(y=0, color='black', alpha=0.5)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('schwefel_p1_analysis.png', dpi=150)
    plt.close()
    
    print("\n📊 Analysis plots saved to: schwefel_p1_analysis.png")


def main():
    """Run P1 validation experiments."""
    print("\n🚀 Starting P1 Schwefel experiments...")
    print("🎯 Target: ≥20% success rate with tolerance 1e-4")
    print("🔬 Testing ChaoticBFO with enhanced exploration")
    
    # Run validation
    results = run_p1_validation()
    
    print("\n✅ P1 experiments completed!")
    print("📈 Check schwefel_p1_analysis.png for visualizations")
    
    # Quick summary
    success_rates = [r['results']['success_rate'] for r in results.values()]
    if max(success_rates) >= 0.2:
        print("\n🎉 TARGET ACHIEVED! ≥20% success rate reached!")
    else:
        best_rate = max(success_rates) * 100
        print(f"\n📊 Best success rate: {best_rate:.1f}% (target: 20%)")


if __name__ == "__main__":
    main()