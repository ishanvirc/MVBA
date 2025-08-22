"""
Reconstruction Quality Ablation - Visualization Script (7 Variants)

This script generates publication-ready visualizations showing how BBRE-inspired
binding mechanisms improve reconstruction quality across all 7 model variants.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def denormalize_image(image_tensor: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """Denormalize image tensor and convert to numpy array."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).numpy()
    
    return image_np


def load_sample_outputs(results_dir: Path, sample_indices: List[int] = None) -> Dict:
    """Load saved outputs for specific samples across all 7 variants."""
    outputs = {}
    
    # All 7 variants
    all_variants = ['baseline', 'spatial_fixed', 'feature_fixed', 'full_fixed', 
                   'spatial', 'feature', 'full']
    
    for variant in all_variants:
        variant_dir = results_dir / 'raw' / variant
        if not variant_dir.exists():
            print(f"Warning: No outputs found for {variant}")
            continue
            
        variant_outputs = []
        if sample_indices is None:
            # Load all available samples
            for output_path in sorted(variant_dir.glob("output_*.pt")):
                variant_outputs.append(torch.load(output_path))
        else:
            # Load specific samples
            for idx in sample_indices:
                output_path = variant_dir / f"output_{idx:06d}.pt"
                if output_path.exists():
                    variant_outputs.append(torch.load(output_path))
        
        outputs[variant] = variant_outputs
    
    return outputs


def create_reconstruction_comparison(outputs: Dict, dataset_info: Dict, save_path: Path):
    """Create side-by-side comparison of reconstructions across all 7 variants."""
    n_samples = min(1, len(outputs.get('baseline', [])))  # Single sample for individual images
    
    # Define order for 2x4 layout
    # Row 1: Original, Baseline, spatial_fixed, spatial
    # Row 2: full, full_fixed, feature, feature_fixed
    row1_variants = [None, 'baseline', 'spatial_fixed', 'spatial']  # None for original
    row1_names = ['Original', 'Baseline', 'spatial_fixed', 'spatial']
    row2_variants = ['full', 'full_fixed', 'feature', 'feature_fixed']
    row2_names = ['full', 'full_fixed', 'feature', 'feature_fixed']
    
    # Setup figure - 2x4 layout with more row separation
    fig = plt.figure(figsize=(16, 10))  
    gs = GridSpec(2, 4, figure=fig, wspace=0.05, hspace=0.25) 
    
    mean = dataset_info['normalization']['mean']
    std = dataset_info['normalization']['std']
    
    # Process single sample
    if 'baseline' in outputs and len(outputs['baseline']) > 0:
        # Get original image from baseline (same for all variants)
        sample_idx = outputs['baseline'][0]['index']
        
        # Load original image
        image_path = Path(dataset_info['test_data_path']) / 'images' / f'image_{sample_idx:06d}.png'
        original = np.array(Image.open(image_path)) / 255.0
        
        # Row 1
        for col in range(4):
            ax = fig.add_subplot(gs[0, col])
            
            if col == 0:  # Original image
                ax.imshow(original)
                ax.set_title(row1_names[col], fontsize=18, pad=8, fontweight='bold')  # Increased from 14 to 18, added bold
            else:  # Variants
                variant = row1_variants[col]
                if variant in outputs and len(outputs[variant]) > 0:
                    recon = outputs[variant][0]['reconstruction']
                    recon_np = denormalize_image(recon, mean, std)
                    ax.imshow(recon_np)
                ax.set_title(row1_names[col], fontsize=18, pad=8, fontweight='bold')  # Increased from 14 to 18, added bold
            
            ax.axis('off')
        
        # Row 2
        for col in range(4):
            ax = fig.add_subplot(gs[1, col])
            
            variant = row2_variants[col]
            if variant in outputs and len(outputs[variant]) > 0:
                recon = outputs[variant][0]['reconstruction']
                recon_np = denormalize_image(recon, mean, std)
                ax.imshow(recon_np)
            
            ax.set_title(row2_names[col], fontsize=18, pad=8, fontweight='bold')  # Increased from 14 to 18, added bold
            ax.axis('off')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def create_per_slot_analysis(outputs: Dict, dataset_info: Dict, save_path: Path):
    """Visualize per-slot reconstructions to show binding quality across 7 variants."""
    import numpy as np
    import matplotlib.gridspec as gridspec
    
    # Use the first (and only) sample in the outputs
    sample_idx = 0
    
    # Create figure with landscape orientation
    fig = plt.figure(figsize=(21, 15), facecolor='white')
    
    # Simple subplot grid
    gs = gridspec.GridSpec(5, 7, figure=fig, 
                          wspace=0.1, hspace=0.1,
                          left=0.08, right=0.96, 
                          top=0.93, bottom=0.07)
    
    mean = dataset_info['normalization']['mean']
    std = dataset_info['normalization']['std']
    
    # Get sample from each variant
    variants = ['baseline', 'spatial_fixed', 'spatial', 'feature_fixed', 'feature', 'full_fixed', 'full']
    variant_names = ['Baseline', 'spatial_fixed', 'spatial', 'feature_fixed', 'feature', 'full_fixed', 'full']
    
    # Process each variant (columns)
    for col, (variant, variant_name) in enumerate(zip(variants, variant_names)):
        if variant not in outputs or sample_idx >= len(outputs[variant]):
            continue
            
        output = outputs[variant][sample_idx]
        
        # Row 0: Original reconstruction
        recon = output['reconstruction']
        recon_np = denormalize_image(recon, mean, std)
        recon_np = np.clip(recon_np, 0, 1)
        
        # Create subplot for reconstruction
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(recon_np)
        ax.set_title(f'{variant_name}', fontsize=22, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Rows 1-4: Per-slot masks
        masks = output['masks']  # (n_slots, H, W)
        n_slots = masks.shape[0]
        
        for slot in range(min(4, n_slots)):
            mask = masks[slot].numpy()
            
            # Create masked reconstruction with high contrast
            # Keep black background but enhance the visible parts
            masked_recon = recon_np * mask[:, :, np.newaxis]
            
            # Enhance contrast for better visibility
            # Scale up the non-zero values to make them more visible
            mask_3d = mask[:, :, np.newaxis]
            enhanced_recon = np.where(mask_3d > 0.01, 
                                     masked_recon * 1.2,  # Brighten the visible parts
                                     masked_recon)
            enhanced_recon = np.clip(enhanced_recon, 0, 1)
            
            # Create subplot for this slot
            ax = fig.add_subplot(gs[slot + 1, col])
            ax.imshow(enhanced_recon)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    # Add row labels on the left (closer to the first column)
    row_labels = ['All Slots', 'Slot 0', 'Slot 1', 'Slot 2', 'Slot 3']
    row_positions = np.linspace(0.85, 0.15, 5)  # Evenly space the labels vertically
    # Move labels closer to the grid (was 0.04, now 0.06)
    label_x = 0.06  # Closer to grid start at 0.08
    for row, (label, pos) in enumerate(zip(row_labels, row_positions)):
        fig.text(label_x, pos, label, 
                rotation=90, ha='center', va='center',
                fontsize=24, fontweight='bold')
    
    # Main title
    fig.suptitle('Per-Slot Reconstruction Analysis', fontsize=28, fontweight='bold', y=1.02)
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_individual_error_heatmaps(outputs: Dict, dataset_info: Dict, save_dir: Path):
    """Create individual error heatmap visualizations for each sample."""
    mean = dataset_info['normalization']['mean']
    std = dataset_info['normalization']['std']
    
    # Get all unique sample indices across variants
    sample_indices = set()
    for variant in outputs:
        for output in outputs[variant]:
            sample_indices.add(output['index'])
    
    print(f"    Creating {len(sample_indices)} individual error heatmaps...")
    
    for idx in sorted(sample_indices):
        # Create figure for this sample
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Sample {idx:06d} - Error Analysis Across Variants', fontsize=14)
        
        # Load original image
        image_path = Path(dataset_info['test_data_path']) / 'images' / f'image_{idx:06d}.png'
        original = np.array(Image.open(image_path)) / 255.0
        
        # Plot original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Get reconstructions from each variant
        variants_to_show = ['baseline', 'spatial_fixed', 'feature_fixed', 'full_fixed', 
                           'spatial', 'feature', 'full']
        variant_names = ['Baseline', 'Spatial (α=1)', 'Feature (α=1)', 'Full (α=1)',
                        'Spatial (BBRE)', 'Feature (BBRE)', 'Full (BBRE)']
        
        col_idx = 1
        row_idx = 0
        
        for variant, variant_name in zip(variants_to_show, variant_names):
            if variant not in outputs:
                continue
                
            # Find this sample in variant outputs
            sample_output = None
            for output in outputs[variant]:
                if output['index'] == idx:
                    sample_output = output
                    break
            
            if sample_output is None:
                continue
            
            # Get current subplot position
            if col_idx > 3:
                row_idx = 1
                col_idx = col_idx - 4
            
            # Get reconstruction
            recon = denormalize_image(sample_output['reconstruction'], mean, std)
            
            # Compute error heatmap
            error = np.mean(np.abs(original - recon), axis=2)
            
            # Plot error heatmap
            im = axes[row_idx, col_idx].imshow(error, cmap='hot', vmin=0, vmax=0.3)
            axes[row_idx, col_idx].set_title(variant_name, fontsize=10)
            axes[row_idx, col_idx].axis('off')
            
            col_idx += 1
        
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Pixel Error', fontsize=10)
        
        # Save individual error heatmap
        save_path = save_dir / f'error_heatmap_sample_{idx:06d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close()

def create_binding_failure_examples(outputs: Dict, metrics: Dict, dataset_info: Dict, save_path: Path):
    """Highlight specific examples where baseline fails but enhanced variants succeed."""
    # Compare baseline vs full model
    if 'baseline' not in metrics or 'full' not in metrics:
        print("Cannot create binding failure examples - missing metrics")
        return
    
    baseline_metrics = {m['index']: m['mse'] for m in metrics['baseline']['metrics']}
    full_metrics = {m['index']: m['mse'] for m in metrics['full']['metrics']}
    
    # Calculate improvements
    improvements = []
    for idx in baseline_metrics:
        if idx in full_metrics:
            improvement = baseline_metrics[idx] - full_metrics[idx]
            improvements.append((idx, improvement, baseline_metrics[idx], full_metrics[idx]))
    
    # Sort by improvement
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    # Get the actual saved indices
    saved_indices = []
    if 'baseline' in outputs:
        saved_indices = [output['index'] for output in outputs['baseline']]
    
    # Filter improvements to only include saved indices
    saved_improvements = [(idx, imp, b_mse, f_mse) for idx, imp, b_mse, f_mse in improvements if idx in saved_indices]
    
    # Use specific indices of test images 253, 531, 763
    specific_indices = [253, 531, 763]
    specific_improvements = []
    for idx in specific_indices:
        for sample_idx, imp, b_mse, f_mse in saved_improvements:
            if sample_idx == idx:
                specific_improvements.append((sample_idx, imp, b_mse, f_mse))
                break
    
    print(f"Using specific samples: {[idx for idx, _, _, _ in specific_improvements]}")
    
    fig = plt.figure(figsize=(22, 11))
    
    # Create custom GridSpec for better control over spacing
    from matplotlib.gridspec import GridSpec
    
    # Create GridSpec with custom width ratios to add more space between certain columns
    gs = GridSpec(3, 6, figure=fig, 
                  width_ratios=[1, 1, 1, 1.2, 1, 1],  # Add extra space after full and Architecture vs BBRE
                  wspace=0.25, hspace=0.25)
    
    axes = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(3)]
    
    mean = dataset_info['normalization']['mean']
    std = dataset_info['normalization']['std']
    
    comparison_variants = ['baseline', 'full_fixed', 'full']
    variant_names = ['Baseline', 'full_fixed', 'full']
    
    shown = 0
    for idx, improvement, baseline_mse, full_mse in specific_improvements:
        # Find samples in outputs
        samples = {}
        for variant in comparison_variants:
            for sample in outputs.get(variant, []):
                if sample['index'] == idx:
                    samples[variant] = sample
                    break
        
        if len(samples) < len(comparison_variants):
            continue
        
        # Load original
        image_path = Path(dataset_info['test_data_path']) / 'images' / f'image_{idx:06d}.png'
        original = np.array(Image.open(image_path)) / 255.0
        
        # Plot original
        axes[shown][0].imshow(original)
        axes[shown][0].set_title('Original' if shown == 0 else '', fontsize=18)
        axes[shown][0].axis('off')
        
        # Add row label (sample indices) to the left of original images
        sample_labels = ['0253', '0531', '0763']
        axes[shown][0].text(-0.15, 0.5, sample_labels[shown], transform=axes[shown][0].transAxes,
                           fontsize=20, fontweight='bold', va='center', ha='right')
        
        # Add label A above first original image (below title)
        if shown == 0:
            axes[shown][0].text(0.05, 1.35, 'A.', transform=axes[shown][0].transAxes,
                               fontsize=20, fontweight='bold', va='top', ha='left')
        
        # Plot comparisons
        for col, (variant, variant_name) in enumerate(zip(comparison_variants, variant_names)):
            if variant in samples:
                recon = denormalize_image(samples[variant]['reconstruction'], mean, std)
                axes[shown][col + 1].imshow(recon)
                axes[shown][col + 1].set_title(variant_name if shown == 0 else '', fontsize=18)
                axes[shown][col + 1].axis('off')
                
                # Calculate MSE for this reconstruction
                mse = F.mse_loss(
                    torch.tensor(original).permute(2, 0, 1),
                    torch.tensor(recon).permute(2, 0, 1)
                ).item()
                
                # Color based on performance
                if variant == 'baseline':
                    color = 'lightcoral'
                elif variant == 'full_fixed':
                    color = 'lightyellow'
                else:  # full
                    color = 'lightgreen'
                
                # Add label B above first baseline (below title)
                if shown == 0 and col == 0:  # baseline column
                    axes[shown][col + 1].text(0.05, 1.35, 'B.', transform=axes[shown][col + 1].transAxes,
                                            fontsize=20, fontweight='bold', va='top', ha='left')
        
        # Plot architecture vs enhancement comparison
        if 'full_fixed' in samples and 'full' in samples:
            fixed_recon = denormalize_image(samples['full_fixed']['reconstruction'], mean, std)
            enhanced_recon = denormalize_image(samples['full']['reconstruction'], mean, std)
            diff = np.abs(fixed_recon - enhanced_recon)
            diff_intensity = np.mean(diff, axis=2)
            
            im1 = axes[shown][4].imshow(diff_intensity, cmap='hot', vmin=0, vmax=0.2)
            axes[shown][4].set_title('Architecture vs BBRE' if shown == 0 else '', fontsize=18)
            
            # Add label C above first Architecture vs BBRE (below title)
            if shown == 0:
                axes[shown][4].text(0.05, 1.35, 'C.', transform=axes[shown][4].transAxes,
                                   fontsize=20, fontweight='bold', va='top', ha='left')
            axes[shown][4].axis('off')
            
            # Add colorbar legend for last row
            if shown == 2:
                from matplotlib.colorbar import Colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(axes[shown][4])
                cax = divider.append_axes("bottom", size="10%", pad=0.15)
                cbar = plt.colorbar(im1, cax=cax, orientation='horizontal')
                cbar.set_label('Pixel-wise Error', fontsize=16, fontweight='bold')
                cbar.ax.tick_params(labelsize=14)
        
        # Plot overall improvement (baseline vs best)
        if 'baseline' in samples and 'full' in samples:
            baseline_recon = denormalize_image(samples['baseline']['reconstruction'], mean, std)
            full_recon = denormalize_image(samples['full']['reconstruction'], mean, std)
            diff = np.abs(baseline_recon - full_recon)
            diff_intensity = np.mean(diff, axis=2)
            
            im2 = axes[shown][5].imshow(diff_intensity, cmap='hot', vmin=0, vmax=0.3)
            axes[shown][5].set_title('Overall Improvement' if shown == 0 else '', fontsize=18)
            axes[shown][5].axis('off')
            
            # Add colorbar legend for last row
            if shown == 2:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(axes[shown][5])
                cax = divider.append_axes("bottom", size="10%", pad=0.15)
                cbar = plt.colorbar(im2, cax=cax, orientation='horizontal')
                cbar.set_label('Pixel-wise Error', fontsize=16, fontweight='bold')
                cbar.ax.tick_params(labelsize=14)
        
        shown += 1
    
    # Remove empty subplots
    for i in range(shown, 3):
        for j in range(6):
            axes[i][j].axis('off')
    
    plt.suptitle('Architecture vs BBRE Enhancement Analysis', fontsize=24, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def create_summary_figure(aggregate_metrics: Dict, save_path: Path):
    """Create summary figure with key metrics across 7 variants."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Order variants: baseline, then fixed alpha, then dynamic alpha
    variant_order = ['baseline', 'spatial_fixed', 'feature_fixed', 'full_fixed',
                    'spatial', 'feature', 'full']
    
    variants = [v for v in variant_order if v in aggregate_metrics]
    
    # Colors: baseline (red), fixed alpha (blues), enhanced (greens)
    colors = ['#ff6b6b', '#87ceeb', '#4682b4', '#1e90ff', '#90ee90', '#32cd32', '#228b22']
    colors = colors[:len(variants)]
    
    # MSE comparison
    mse_values = [aggregate_metrics[v]['mse'] for v in variants]
    mse_stds = [aggregate_metrics[v]['mse_std'] for v in variants]
    
    bars1 = axes[0, 0].bar(range(len(variants)), mse_values, yerr=mse_stds, 
                          capsize=5, color=colors, alpha=0.8)
    axes[0, 0].set_title('Mean Squared Error (Lower is Better)', fontsize=14, pad=10)
    axes[0, 0].set_ylabel('MSE', fontsize=12)
    axes[0, 0].set_xticks(range(len(variants)))
    axes[0, 0].set_xticklabels([v.replace('_', '\n') for v in variants], rotation=45, ha='right', fontsize=10)
    
    # Add value labels
    for bar, val in zip(bars1, mse_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # PSNR comparison
    psnr_values = [aggregate_metrics[v]['psnr'] for v in variants]
    psnr_stds = [aggregate_metrics[v]['psnr_std'] for v in variants]
    
    bars2 = axes[0, 1].bar(range(len(variants)), psnr_values, yerr=psnr_stds,
                          capsize=5, color=colors, alpha=0.8)
    axes[0, 1].set_title('PSNR (Higher is Better)', fontsize=14, pad=10)
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 1].set_xticks(range(len(variants)))
    axes[0, 1].set_xticklabels([v.replace('_', '\n') for v in variants], rotation=45, ha='right', fontsize=10)
    
    # SSIM comparison
    ssim_values = [aggregate_metrics[v]['ssim'] for v in variants]
    ssim_stds = [aggregate_metrics[v]['ssim_std'] for v in variants]
    
    bars3 = axes[1, 0].bar(range(len(variants)), ssim_values, yerr=ssim_stds,
                          capsize=5, color=colors, alpha=0.8)
    axes[1, 0].set_title('SSIM (Higher is Better)', fontsize=14, pad=10)
    axes[1, 0].set_ylabel('SSIM', fontsize=12)
    axes[1, 0].set_xticks(range(len(variants)))
    axes[1, 0].set_xticklabels([v.replace('_', '\n') for v in variants], rotation=45, ha='right', fontsize=10)
    axes[1, 0].set_ylim(0, 1)
    
    # Architecture vs Enhancement comparison
    if 'baseline' in aggregate_metrics:
        baseline_mse = aggregate_metrics['baseline']['mse']
        
        # Compare architecture contributions and BBRE effects
        comparisons = []
        labels = []
        comparison_colors = []
        
        # Architecture only (fixed alpha)
        for variant in ['spatial_fixed', 'feature_fixed', 'full_fixed']:
            if variant in aggregate_metrics:
                variant_mse = aggregate_metrics[variant]['mse']
                improvement = (baseline_mse - variant_mse) / baseline_mse * 100
                comparisons.append(improvement)
                labels.append(f'{variant.replace("_fixed", "")}\n(arch only)')
                comparison_colors.append('#4682b4')
        
        # BBRE enhancement effects
        enhancement_pairs = [('spatial_fixed', 'spatial'), ('feature_fixed', 'feature'), ('full_fixed', 'full')]
        for fixed_var, enhanced_var in enhancement_pairs:
            if fixed_var in aggregate_metrics and enhanced_var in aggregate_metrics:
                fixed_mse = aggregate_metrics[fixed_var]['mse']
                enhanced_mse = aggregate_metrics[enhanced_var]['mse']
                improvement = (fixed_mse - enhanced_mse) / fixed_mse * 100
                comparisons.append(improvement)
                labels.append(f'{enhanced_var}\n(BBRE effect)')
                comparison_colors.append('#32cd32')
        
        bars4 = axes[1, 1].bar(range(len(comparisons)), comparisons, color=comparison_colors, alpha=0.8)
        axes[1, 1].set_title('Architecture vs BBRE Enhancement', fontsize=14, pad=10)
        axes[1, 1].set_ylabel('Improvement (%)', fontsize=12)
        axes[1, 1].set_xticks(range(len(comparisons)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars4, comparisons):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Reconstruction Quality Metrics Summary (7 Variants)', fontsize=18, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for reconstruction quality')
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
    
    # Add test data path to dataset info
    dataset_info = {
        'normalization': {
            'mean': [0.485, 0.456, 0.406],  # Default ImageNet normalization
            'std': [0.229, 0.224, 0.225]
        },
        'test_data_path': config['data']['test_data_path']
    }
    
    # Load actual normalization stats
    with open(config['data']['dataset_info_path'], 'r') as f:
        actual_dataset_info = json.load(f)
        dataset_info['normalization'] = actual_dataset_info['normalization']
    
    results_dir = Path(__file__).parent / args.results_dir
    vis_dir = Path(__file__).parent.parent / 'visualizations'
    
    # Load metrics
    with open(results_dir / 'metrics' / 'per_sample_metrics.json', 'r') as f:
        per_sample_metrics = json.load(f)
    
    with open(results_dir / 'metrics' / 'aggregate_metrics.json', 'r') as f:
        aggregate_metrics = json.load(f)
    
    # Load sample outputs
    print("Loading sample outputs...")
    sample_indices = config['evaluation']['sample_indices']
    outputs = load_sample_outputs(results_dir, sample_indices)
    
    # For binding improvements, load ALL saved samples
    print("Loading all saved outputs for binding analysis...")
    all_outputs = load_sample_outputs(results_dir, None)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Reconstruction comparison
    print("  Creating reconstruction comparisons...")
    vis_dir_comp = vis_dir / 'reconstruction_comparison'
    vis_dir_comp.mkdir(parents=True, exist_ok=True)
    
    # Create individual reconstruction comparisons for each sample
    for i, sample_idx in enumerate(sample_indices):
        print(f"    Creating reconstruction comparison for sample {sample_idx}...")
        # Get outputs for this specific sample
        sample_outputs = {}
        for variant, variant_outputs in outputs.items():
            for output in variant_outputs:
                if output['index'] == sample_idx:
                    sample_outputs[variant] = [output]
                    break
        
        if len(sample_outputs) > 0:
            create_reconstruction_comparison(sample_outputs, dataset_info,
                                           vis_dir_comp / f'reconstruction_comparison_sample_{sample_idx:06d}.png')
    
    # 2. Per-slot analysis
    print("  Creating per-slot analyses...")
    vis_dir_slot = vis_dir / 'per_slot_analysis'
    vis_dir_slot.mkdir(parents=True, exist_ok=True)
    
    # Create individual per-slot analyses for each sample
    for i, sample_idx in enumerate(sample_indices):
        print(f"    Creating per-slot analysis for sample {sample_idx}...")
        # Get outputs for this specific sample
        sample_outputs = {}
        for variant, variant_outputs in outputs.items():
            for j, output in enumerate(variant_outputs):
                if output['index'] == sample_idx:
                    sample_outputs[variant] = [output]
                    break
        
        if len(sample_outputs) > 0:
            create_per_slot_analysis(sample_outputs, dataset_info,
                                   vis_dir_slot / f'per_slot_analysis_sample_{sample_idx:06d}.png')
    
    # 3. Individual error heatmaps for each sample
    print("  Creating individual error heatmaps...")
    vis_dir_error = vis_dir / 'error_heatmaps'
    vis_dir_error.mkdir(parents=True, exist_ok=True)
    create_individual_error_heatmaps(outputs, dataset_info, vis_dir_error)
    
    # 4. Binding failure examples (summary)
    print("  Creating binding failure examples...")
    create_binding_failure_examples(all_outputs, per_sample_metrics, dataset_info,
                                  vis_dir_error / 'binding_improvements_summary.png')
    
    # 5. Summary figure
    print("  Creating summary figure...")
    vis_dir_summary = vis_dir / 'summary_figures'
    vis_dir_summary.mkdir(parents=True, exist_ok=True)
    create_summary_figure(aggregate_metrics,
                        vis_dir_summary / 'metrics_summary.png')
    
    print(f"\nVisualizations saved to {vis_dir}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS (7-VARIANT ANALYSIS)")
    print("="*80)
    
    variants = list(aggregate_metrics.keys())
    print(f"\nAnalyzed variants: {', '.join(variants)}")
    
    if 'baseline' in aggregate_metrics:
        baseline_mse = aggregate_metrics['baseline']['mse']
        
        # Architecture-only improvements
        print("\nArchitecture-Only Contributions (alpha = 1):")
        for variant in ['spatial_fixed', 'feature_fixed', 'full_fixed']:
            if variant in aggregate_metrics:
                variant_mse = aggregate_metrics[variant]['mse']
                improvement = (baseline_mse - variant_mse) / baseline_mse * 100
                print(f"  {variant}: {improvement:.1f}% improvement over baseline")
        
        # BBRE enhancement effects
        print("\nBBRE Enhancement Effects:")
        enhancement_pairs = [('spatial_fixed', 'spatial'), ('feature_fixed', 'feature'), ('full_fixed', 'full')]
        for fixed_var, enhanced_var in enhancement_pairs:
            if fixed_var in aggregate_metrics and enhanced_var in aggregate_metrics:
                fixed_mse = aggregate_metrics[fixed_var]['mse']
                enhanced_mse = aggregate_metrics[enhanced_var]['mse']
                improvement = (fixed_mse - enhanced_mse) / fixed_mse * 100
                print(f"  {enhanced_var} vs {fixed_var}: {improvement:.1f}% additional improvement from BBRE")
        
        # Overall best performance
        if 'full' in aggregate_metrics:
            full_mse = aggregate_metrics['full']['mse']
            total_improvement = (baseline_mse - full_mse) / baseline_mse * 100
            print(f"\nOverall: Full model shows {total_improvement:.1f}% improvement over baseline")
            print("This demonstrates both architectural and enhancement contributions!")


if __name__ == '__main__':
    main()