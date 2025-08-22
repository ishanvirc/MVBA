#!/usr/bin/env python3
"""
Reconstruction Quality Ablation - Analysis Script (7 Variants)

This script analyzes the evaluation results to identify binding improvements
and perform statistical comparisons between all 7 model variants.
Performs 21 pairwise comparisons (7 choose 2).
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def load_results(results_dir: Path) -> Dict:
    """Load evaluation results from saved files."""
    # Load per-sample metrics
    with open(results_dir / 'metrics' / 'per_sample_metrics.json', 'r') as f:
        per_sample = json.load(f)
    
    # Load aggregate metrics
    with open(results_dir / 'metrics' / 'aggregate_metrics.json', 'r') as f:
        aggregate = json.load(f)
    
    return per_sample, aggregate


def perform_statistical_tests(per_sample_results: Dict, metric: str = 'mse') -> Dict:
    """Perform paired t-tests between model variants."""
    variants = list(per_sample_results.keys())
    n_variants = len(variants)
    
    # Extract metric values for each variant
    metric_values = {}
    for variant in variants:
        metric_values[variant] = [m[metric] for m in per_sample_results[variant]['metrics']]
    
    # Perform pairwise comparisons
    comparisons = {}
    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            variant1, variant2 = variants[i], variants[j]
            values1 = metric_values[variant1]
            values2 = metric_values[variant2]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(values1, values2)
            
            # Calculate effect size (Cohen's d)
            diff = np.array(values1) - np.array(values2)
            cohen_d = np.mean(diff) / np.std(diff)
            
            # Calculate percentage improvement with corrected formula
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            if metric in ['mse', 'l1']:  # Lower is better
                # Positive improvement = reduction in error
                if mean1 == 0:
                    improvement = 0 if mean2 == 0 else float('-inf')
                else:
                    improvement = (mean1 - mean2) / abs(mean1) * 100
            else:  # Higher is better (psnr, ssim)
                # Positive improvement = increase in quality
                if mean1 == 0:
                    improvement = float('inf') if mean2 > 0 else (0 if mean2 == 0 else float('-inf'))
                else:
                    improvement = (mean2 - mean1) / abs(mean1) * 100
            
            comparisons[f"{variant1}_vs_{variant2}"] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohen_d': float(cohen_d),
                'mean_diff': float(mean1 - mean2),
                'improvement_percent': float(improvement),
                'significant': bool(p_value < 0.05)
            }
    
    return comparisons


def create_comparison_matrix(aggregate_results: Dict, metric: str = 'mse') -> pd.DataFrame:
    """Create a comparison matrix showing relative improvements."""
    variants = list(aggregate_results.keys())
    n_variants = len(variants)
    
    # Create matrix
    matrix = np.zeros((n_variants, n_variants))
    
    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants):
            if i == j:
                matrix[i, j] = 0
            else:
                val1 = aggregate_results[var1][metric]
                val2 = aggregate_results[var2][metric]
                if metric in ['mse', 'l1']:  # Lower is better
                    improvement = (val1 - val2) / abs(val1) * 100 if val1 != 0 else 0
                else:  # Higher is better
                    improvement = (val2 - val1) / abs(val1) * 100 if val1 != 0 else 0
                matrix[i, j] = improvement
    
    # Create DataFrame
    df = pd.DataFrame(matrix, index=variants, columns=variants)
    return df


def analyze_binding_improvements(per_sample_results: Dict) -> Dict:
    """Analyze specific improvements in binding quality across all 7 variants."""
    analysis = {}
    
    # Compare baseline vs full model (if both exist)
    if 'baseline' in per_sample_results and 'full' in per_sample_results:
        baseline_mse = [m['mse'] for m in per_sample_results['baseline']['metrics']]
        full_mse = [m['mse'] for m in per_sample_results['full']['metrics']]
        
        # Find samples with largest improvements
        improvements = np.array(baseline_mse) - np.array(full_mse)
        top_improved_indices = np.argsort(improvements)[::-1][:10]
        
        analysis['top_improvements'] = [
            {
                'index': per_sample_results['baseline']['metrics'][idx]['index'],
                'baseline_mse': baseline_mse[idx],
                'full_mse': full_mse[idx],
                'improvement': improvements[idx]
            }
            for idx in top_improved_indices
        ]
    
    # Architecture vs Enhancement Analysis (7-variant specific)
    architecture_analysis = {}
    
    # Architecture-only improvements (fixed alpha = 1)
    if 'baseline' in per_sample_results:
        baseline_mse = np.mean([m['mse'] for m in per_sample_results['baseline']['metrics']])
        
        for variant in ['spatial_fixed', 'feature_fixed', 'full_fixed']:
            if variant in per_sample_results:
                variant_mse = np.mean([m['mse'] for m in per_sample_results[variant]['metrics']])
                architecture_analysis[f'{variant}_improvement'] = baseline_mse - variant_mse
    
    # BBRE Enhancement effects
    enhancement_analysis = {}
    
    enhancement_pairs = [
        ('spatial_fixed', 'spatial'),
        ('feature_fixed', 'feature'),
        ('full_fixed', 'full')
    ]
    
    for fixed_variant, enhanced_variant in enhancement_pairs:
        if fixed_variant in per_sample_results and enhanced_variant in per_sample_results:
            fixed_mse = np.mean([m['mse'] for m in per_sample_results[fixed_variant]['metrics']])
            enhanced_mse = np.mean([m['mse'] for m in per_sample_results[enhanced_variant]['metrics']])
            enhancement_analysis[f'{enhanced_variant}_bbre_effect'] = fixed_mse - enhanced_mse
    
    analysis['architecture_contributions'] = architecture_analysis
    analysis['bbre_enhancements'] = enhancement_analysis
    
    # Traditional component analysis
    if all(v in per_sample_results for v in ['baseline', 'spatial', 'feature', 'full']):
        baseline_mse = np.mean([m['mse'] for m in per_sample_results['baseline']['metrics']])
        spatial_mse = np.mean([m['mse'] for m in per_sample_results['spatial']['metrics']])
        feature_mse = np.mean([m['mse'] for m in per_sample_results['feature']['metrics']])
        full_mse = np.mean([m['mse'] for m in per_sample_results['full']['metrics']])
        
        # Calculate individual contributions
        spatial_contrib = baseline_mse - spatial_mse
        feature_contrib = baseline_mse - feature_mse
        full_improvement = baseline_mse - full_mse
        synergy = full_improvement - (spatial_contrib + feature_contrib)
        
        analysis['component_contributions'] = {
            'spatial_contribution': float(spatial_contrib),
            'feature_contribution': float(feature_contrib),
            'synergistic_effect': float(synergy),
            'total_improvement': float(full_improvement)
        }
    
    return analysis


def save_analysis_report(results_dir: Path, all_results: Dict):
    """Save comprehensive analysis report."""
    report_path = results_dir / 'analysis' / 'analysis_report.txt'
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RECONSTRUCTION QUALITY ABLATION - ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Aggregate metrics
        f.write("1. AGGREGATE METRICS\n")
        f.write("-"*40 + "\n")
        for variant, metrics in all_results['aggregate'].items():
            f.write(f"\n{variant.upper()}:\n")
            for metric, value in metrics.items():
                if '_std' not in metric:
                    std_key = f"{metric}_std"
                    if std_key in metrics:
                        f.write(f"  {metric.upper()}: {value:.6f} ± {metrics[std_key]:.6f}\n")
                    else:
                        f.write(f"  {metric.upper()}: {value:.6f}\n")
        
        # Statistical comparisons
        f.write("\n\n2. STATISTICAL COMPARISONS (MSE)\n")
        f.write("-"*40 + "\n")
        for comparison, results in all_results['statistical_tests'].items():
            f.write(f"\n{comparison}:\n")
            f.write(f"  Mean difference: {results['mean_diff']:.6f}\n")
            f.write(f"  Improvement: {results['improvement_percent']:.2f}%\n")
            f.write(f"  p-value: {results['p_value']:.4f}")
            f.write(f" ({'SIGNIFICANT' if results['significant'] else 'not significant'})\n")
            f.write(f"  Effect size (Cohen's d): {results['cohen_d']:.3f}\n")
        
        # Architecture vs Enhancement Analysis
        if 'binding_analysis' in all_results:
            binding = all_results['binding_analysis']
            
            if 'architecture_contributions' in binding:
                f.write("\n\n3. ARCHITECTURE-ONLY CONTRIBUTIONS (alpha=1)\n")
                f.write("-"*40 + "\n")
                for contrib, value in binding['architecture_contributions'].items():
                    f.write(f"{contrib}: {value:.6f}\n")
            
            if 'bbre_enhancements' in binding:
                f.write("\n\n4. BBRE ENHANCEMENT EFFECTS\n")
                f.write("-"*40 + "\n")
                for effect, value in binding['bbre_enhancements'].items():
                    f.write(f"{effect}: {value:.6f}\n")
            
            # Traditional component analysis
            if 'component_contributions' in binding:
                f.write("\n\n5. TRADITIONAL COMPONENT CONTRIBUTIONS\n")
                f.write("-"*40 + "\n")
                contrib = binding['component_contributions']
                f.write(f"Spatial binding contribution: {contrib['spatial_contribution']:.6f}\n")
                f.write(f"Feature binding contribution: {contrib['feature_contribution']:.6f}\n")
                f.write(f"Synergistic effect: {contrib['synergistic_effect']:.6f}\n")
                f.write(f"Total improvement: {contrib['total_improvement']:.6f}\n")
        
        # Top improvements
        if 'binding_analysis' in all_results and 'top_improvements' in all_results['binding_analysis']:
            f.write("\n\n6. TOP IMPROVED SAMPLES\n")
            f.write("-"*40 + "\n")
            f.write("Sample Index | Baseline MSE | Full MSE | Improvement\n")
            for sample in all_results['binding_analysis']['top_improvements'][:5]:
                f.write(f"{sample['index']:12d} | {sample['baseline_mse']:12.6f} | "
                       f"{sample['full_mse']:8.6f} | {sample['improvement']:11.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze reconstruction quality results')
    parser.add_argument('--results-dir', type=str, 
                        default='../results',
                        help='Path to results directory')
    parser.add_argument('--config', type=str,
                        default='../config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load results
    results_dir = Path(__file__).parent / args.results_dir
    per_sample_results, aggregate_results = load_results(results_dir)
    
    print("Loaded results for variants:", list(per_sample_results.keys()))
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    statistical_tests = {}
    for metric in ['mse', 'psnr', 'ssim']:
        print(f"  Testing {metric}...")
        statistical_tests[metric] = perform_statistical_tests(per_sample_results, metric)
    
    # Create comparison matrices
    print("\nCreating comparison matrices...")
    comparison_matrices = {}
    for metric in ['mse', 'psnr', 'ssim']:
        comparison_matrices[metric] = create_comparison_matrix(aggregate_results, metric)
    
    # Analyze binding improvements
    print("\nAnalyzing binding improvements...")
    binding_analysis = analyze_binding_improvements(per_sample_results)
    
    # Save all results
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    # Save statistical tests
    with open(analysis_dir / 'statistical_tests.json', 'w') as f:
        json.dump(statistical_tests, f, indent=2)
    
    # Save comparison matrices
    for metric, matrix in comparison_matrices.items():
        matrix.to_csv(analysis_dir / f'comparison_matrix_{metric}.csv')
    
    # Save binding analysis
    with open(analysis_dir / 'binding_analysis.json', 'w') as f:
        json.dump(binding_analysis, f, indent=2)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: MSE comparison
    variants = list(aggregate_results.keys())
    mse_means = [aggregate_results[v]['mse'] for v in variants]
    mse_stds = [aggregate_results[v]['mse_std'] for v in variants]
    
    axes[0, 0].bar(variants, mse_means, yerr=mse_stds, capsize=5)
    axes[0, 0].set_title('MSE Comparison')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: PSNR comparison
    psnr_means = [aggregate_results[v]['psnr'] for v in variants]
    psnr_stds = [aggregate_results[v]['psnr_std'] for v in variants]
    
    axes[0, 1].bar(variants, psnr_means, yerr=psnr_stds, capsize=5)
    axes[0, 1].set_title('PSNR Comparison')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Improvement heatmap
    mse_matrix = comparison_matrices['mse']
    sns.heatmap(mse_matrix, annot=True, fmt='.1f', cmap='RdBu_r', 
                center=0, ax=axes[1, 0], cbar_kws={'label': 'Improvement %'})
    axes[1, 0].set_title('MSE Improvement Matrix (%)')
    
    # Plot 4: Component contributions (if available)
    if 'component_contributions' in binding_analysis:
        contrib = binding_analysis['component_contributions']
        components = ['Spatial', 'Feature', 'Synergy']
        values = [contrib['spatial_contribution'], 
                 contrib['feature_contribution'],
                 contrib['synergistic_effect']]
        
        axes[1, 1].bar(components, values)
        axes[1, 1].set_title('Component Contributions to MSE Reduction')
        axes[1, 1].set_ylabel('MSE Reduction')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(analysis_dir / 'analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive report
    all_results = {
        'aggregate': aggregate_results,
        'statistical_tests': statistical_tests['mse'],  # Focus on MSE for report
        'binding_analysis': binding_analysis,
        'comparison_matrices': {k: v.to_dict() for k, v in comparison_matrices.items()}
    }
    save_analysis_report(results_dir, all_results)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Print key improvements for 7-variant analysis
    variants = list(aggregate_results.keys())
    print(f"\nVariants analyzed: {', '.join(variants)}")
    print(f"Total pairwise comparisons: {len(statistical_tests['mse'])}")
    
    # Key architecture vs enhancement comparisons
    if 'baseline' in aggregate_results and 'full' in aggregate_results:
        baseline_mse = aggregate_results['baseline']['mse']
        full_mse = aggregate_results['full']['mse']
        improvement = (baseline_mse - full_mse) / baseline_mse * 100
        print(f"\nBaseline → Full Model (Overall):")
        print(f"  MSE: {baseline_mse:.6f} → {full_mse:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    # Architecture-only effects
    architecture_pairs = [('baseline', 'spatial_fixed'), ('baseline', 'feature_fixed'), ('baseline', 'full_fixed')]
    for variant1, variant2 in architecture_pairs:
        if variant1 in aggregate_results and variant2 in aggregate_results:
            mse1 = aggregate_results[variant1]['mse']
            mse2 = aggregate_results[variant2]['mse']
            improvement = (mse1 - mse2) / mse1 * 100
            print(f"\n{variant1} → {variant2} (Architecture Only):")
            print(f"  MSE: {mse1:.6f} → {mse2:.6f}  ({improvement:+.1f}%)")
    
    # BBRE enhancement effects
    enhancement_pairs = [('spatial_fixed', 'spatial'), ('feature_fixed', 'feature'), ('full_fixed', 'full')]
    for variant1, variant2 in enhancement_pairs:
        if variant1 in aggregate_results and variant2 in aggregate_results:
            mse1 = aggregate_results[variant1]['mse']
            mse2 = aggregate_results[variant2]['mse']
            improvement = (mse1 - mse2) / mse1 * 100
            print(f"\n{variant1} → {variant2} (BBRE Enhancement):")
            print(f"  MSE: {mse1:.6f} → {mse2:.6f}  ({improvement:+.1f}%)")
    
    print(f"\nAnalysis complete. Results saved to {analysis_dir}")


if __name__ == '__main__':
    main()