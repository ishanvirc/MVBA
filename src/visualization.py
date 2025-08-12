"""
Comprehensive Visualization Tools for MVBA Architecture

This module provides extensive visualization capabilities for debugging and analyzing
binding behavior in the Minimal Viable Binding Architecture (MVBA). The tools support
both real-time training visualization and detailed post-hoc analysis.

Key visualization categories:
1. Binding Visualization - Spatial attention maps, slot assignments, feature binding
2. Training Progress - Loss curves, metrics evolution, curriculum learning
3. Comparative Analysis - Model comparisons, ablation studies, before/after
4. Interactive Analysis - Jupyter-friendly widgets, matplotlib interactions
5. Diagnostic Tools - Attention patterns, slot utilization, failure modes

The implementation is designed to be:
- Memory efficient for large datasets
- Compatible with both training and inference
- Customizable for different analysis needs
- Publication-ready output quality
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from collections import defaultdict
import logging

try:
    from IPython.display import display, HTML
    from ipywidgets import interact, interactive, IntSlider, FloatSlider, Dropdown
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    warnings.warn("Jupyter widgets not available, interactive features disabled")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available, some interactive plots disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    save_format: str = 'png'
    color_scheme: str = 'viridis'
    font_size: int = 12
    line_width: float = 2.0
    alpha_transparency: float = 0.7
    grid_alpha: float = 0.3
    save_path: Optional[str] = None
    interactive: bool = False
    high_quality: bool = False


class MVBAVisualizer:
    """
    Comprehensive visualization suite for MVBA model analysis.
    
    This class provides all visualization capabilities needed for debugging,
    analyzing, and presenting MVBA model behavior and performance.
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize MVBA visualizer.
        
        Args:
            config: Visualization configuration
            device: Device for tensor operations
        """
        self.config = config or VisualizationConfig()
        self.device = device or torch.device('cpu')
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'lines.linewidth': self.config.line_width,
            'grid.alpha': self.config.grid_alpha,
            'figure.dpi': self.config.dpi
        })
        
        # Initialize color schemes
        self._setup_color_schemes()
        
        # Track visualization state for interactive use
        self.current_data = {}
        self.plot_history = []
    
    def _setup_color_schemes(self):
        """Setup custom color schemes for different visualization types."""
        # Custom colormap for attention visualization
        colors_attention = ['#000033', '#000055', '#0000ff', '#3333ff', '#6666ff', '#9999ff', '#ccccff', '#ffffff']
        self.cmap_attention = LinearSegmentedColormap.from_list('attention', colors_attention)
        
        # Custom colormap for slot assignments
        self.colors_slots = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        
        # Custom colormap for alpha values
        colors_alpha = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026']
        self.cmap_alpha = LinearSegmentedColormap.from_list('alpha', colors_alpha)
    
    # ===============================
    # 1. BINDING VISUALIZATION
    # ===============================
    
    def plot_spatial_attention_overlay(
        self,
        images: torch.Tensor,
        spatial_attention: torch.Tensor,
        save_path: Optional[str] = None,
        batch_indices: Optional[List[int]] = None,
        slot_indices: Optional[List[int]] = None,
        overlay_alpha: float = 0.6,
        show_individual_slots: bool = True,
        show_combined: bool = True
    ) -> plt.Figure:
        """
        Plot spatial attention maps overlaid on input images.
        
        Args:
            images: Input images (B, C, H, W)
            spatial_attention: Spatial attention maps (B, n_slots, H, W)
            save_path: Optional path to save the figure
            batch_indices: Which batch samples to visualize
            slot_indices: Which slots to visualize
            overlay_alpha: Transparency of attention overlay
            show_individual_slots: Whether to show individual slot attention
            show_combined: Whether to show combined attention
            
        Returns:
            Matplotlib figure object
        """
        B, C, H, W = images.shape
        _, n_slots, _, _ = spatial_attention.shape
        
        # Select samples and slots to visualize
        if batch_indices is None:
            batch_indices = list(range(min(4, B)))
        if slot_indices is None:
            slot_indices = list(range(n_slots))
        
        n_batch = len(batch_indices)
        n_show_slots = len(slot_indices)
        
        # Calculate grid layout
        if show_individual_slots and show_combined:
            n_cols = n_show_slots + 2  # Original + individual slots + combined
        elif show_individual_slots:
            n_cols = n_show_slots + 1  # Original + individual slots
        else:
            n_cols = 2  # Original + combined
        
        # Create figure
        fig, axes = plt.subplots(
            n_batch, n_cols,
            figsize=(n_cols * 3, n_batch * 3),
            squeeze=False
        )
        
        for i, batch_idx in enumerate(batch_indices):
            # Convert image to numpy and denormalize
            img = images[batch_idx].cpu()
            if C == 1:
                img_np = img.squeeze(0).numpy()
                cmap_img = 'gray'
            else:
                img_np = img.permute(1, 2, 0).numpy()
                # Clamp to [0, 1] range
                img_np = np.clip(img_np, 0, 1)
                cmap_img = None
            
            col_idx = 0
            
            # Show original image
            axes[i, col_idx].imshow(img_np, cmap=cmap_img)
            axes[i, col_idx].set_title(f'Original (Batch {batch_idx})')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Show individual slot attention
            if show_individual_slots:
                for slot_idx in slot_indices:
                    attention = spatial_attention[batch_idx, slot_idx].cpu().numpy()
                    
                    # Create overlay
                    axes[i, col_idx].imshow(img_np, cmap=cmap_img)
                    im = axes[i, col_idx].imshow(
                        attention,
                        alpha=overlay_alpha,
                        cmap=self.cmap_attention,
                        vmin=0,
                        vmax=attention.max()
                    )
                    axes[i, col_idx].set_title(f'Slot {slot_idx}')
                    axes[i, col_idx].axis('off')
                    
                    # Add colorbar for first row
                    if i == 0:
                        cbar = plt.colorbar(im, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
                        cbar.set_label('Attention', rotation=270, labelpad=15)
                    
                    col_idx += 1
            
            # Show combined attention
            if show_combined:
                combined_attention = spatial_attention[batch_idx].sum(dim=0).cpu().numpy()
                
                axes[i, col_idx].imshow(img_np, cmap=cmap_img)
                im = axes[i, col_idx].imshow(
                    combined_attention,
                    alpha=overlay_alpha,
                    cmap=self.cmap_attention,
                    vmin=0,
                    vmax=combined_attention.max()
                )
                axes[i, col_idx].set_title('Combined Attention')
                axes[i, col_idx].axis('off')
                
                # Add colorbar for first row
                if i == 0:
                    cbar = plt.colorbar(im, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
                    cbar.set_label('Combined Attention', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_slot_assignments(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        spatial_attention: torch.Tensor,
        reconstructions: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        batch_indices: Optional[List[int]] = None,
        show_reconstructions: bool = False
    ) -> plt.Figure:
        """
        Visualize slot assignments and object segmentations.
        
        Args:
            images: Input images (B, C, H, W)
            masks: Object masks (B, n_slots, H, W)
            spatial_attention: Spatial attention maps (B, n_slots, H, W)
            reconstructions: Optional slot reconstructions (B, n_slots, C, H, W)
            save_path: Optional path to save the figure
            batch_indices: Which batch samples to visualize
            show_reconstructions: Whether to show individual slot reconstructions
            
        Returns:
            Matplotlib figure object
        """
        B, C, H, W = images.shape
        _, n_slots, _, _ = masks.shape
        
        if batch_indices is None:
            batch_indices = list(range(min(4, B)))
        
        n_batch = len(batch_indices)
        
        # Calculate number of columns
        n_cols = 3  # Original, masks, attention
        if show_reconstructions and reconstructions is not None:
            n_cols += n_slots
        
        # Create figure
        fig, axes = plt.subplots(
            n_batch, n_cols,
            figsize=(n_cols * 2.5, n_batch * 2.5),
            squeeze=False
        )
        
        for i, batch_idx in enumerate(batch_indices):
            img = images[batch_idx].cpu()
            
            # Convert image for display
            if C == 1:
                img_np = img.squeeze(0).numpy()
                cmap_img = 'gray'
            else:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                cmap_img = None
            
            col_idx = 0
            
            # Original image
            axes[i, col_idx].imshow(img_np, cmap=cmap_img)
            axes[i, col_idx].set_title(f'Original (Batch {batch_idx})')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Slot assignments (colored masks)
            mask_colored = self._create_colored_segmentation(masks[batch_idx])
            axes[i, col_idx].imshow(mask_colored)
            axes[i, col_idx].set_title('Slot Assignments')
            axes[i, col_idx].axis('off')
            
            # Add legend for slots
            if i == 0:
                for slot_idx in range(n_slots):
                    color = self.colors_slots[slot_idx % len(self.colors_slots)]
                    axes[i, col_idx].plot([], [], 's', color=color, 
                                        label=f'Slot {slot_idx}', markersize=8)
                axes[i, col_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            col_idx += 1
            
            # Combined spatial attention
            combined_attention = spatial_attention[batch_idx].sum(dim=0).cpu().numpy()
            im = axes[i, col_idx].imshow(combined_attention, cmap=self.cmap_attention)
            axes[i, col_idx].set_title('Spatial Attention')
            axes[i, col_idx].axis('off')
            
            if i == 0:
                cbar = plt.colorbar(im, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
                cbar.set_label('Attention', rotation=270, labelpad=15)
            col_idx += 1
            
            # Individual slot reconstructions
            if show_reconstructions and reconstructions is not None:
                for slot_idx in range(n_slots):
                    recon = reconstructions[batch_idx, slot_idx].cpu()
                    
                    if C == 1:
                        recon_np = recon.squeeze(0).numpy()
                    else:
                        recon_np = recon.permute(1, 2, 0).numpy()
                        recon_np = np.clip(recon_np, 0, 1)
                    
                    axes[i, col_idx].imshow(recon_np, cmap=cmap_img)
                    axes[i, col_idx].set_title(f'Slot {slot_idx} Recon')
                    axes[i, col_idx].axis('off')
                    col_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_feature_binding_heatmaps(
        self,
        bound_features: torch.Tensor,
        alphas: Dict[str, torch.Tensor],
        slot_similarities: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        batch_indices: Optional[List[int]] = None
    ) -> plt.Figure:
        """
        Visualize feature binding through heatmaps and alpha distributions.
        
        Args:
            bound_features: Bound feature representations (B, n_slots, slot_dim)
            alphas: Alpha values dict with 'spatial' and 'feature' keys
            slot_similarities: Optional similarity matrix between slots
            save_path: Optional path to save the figure
            batch_indices: Which batch samples to visualize
            
        Returns:
            Matplotlib figure object
        """
        B, n_slots, slot_dim = bound_features.shape
        
        if batch_indices is None:
            batch_indices = list(range(min(2, B)))
        
        n_batch = len(batch_indices)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4 * n_batch))
        gs = GridSpec(n_batch, 4, figure=fig)
        
        for i, batch_idx in enumerate(batch_indices):
            # Feature similarity matrix
            ax1 = fig.add_subplot(gs[i, 0])
            features = bound_features[batch_idx].cpu().numpy()  # (n_slots, slot_dim)
            
            # Compute cosine similarity
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = np.dot(features_norm, features_norm.T)
            
            im1 = ax1.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax1.set_title(f'Feature Similarity (B{batch_idx})')
            ax1.set_xlabel('Slot Index')
            ax1.set_ylabel('Slot Index')
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Cosine Similarity')
            
            # Add text annotations
            for row in range(n_slots):
                for col in range(n_slots):
                    text = ax1.text(col, row, f'{similarity_matrix[row, col]:.2f}',
                                  ha="center", va="center", color="black" if abs(similarity_matrix[row, col]) < 0.5 else "white")
            
            # Feature magnitude heatmap
            ax2 = fig.add_subplot(gs[i, 1])
            feature_magnitudes = np.linalg.norm(features, axis=1)
            feature_heatmap = features / (feature_magnitudes[:, np.newaxis] + 1e-8)
            
            im2 = ax2.imshow(feature_heatmap[:, :min(50, slot_dim)], cmap='viridis', aspect='auto')
            ax2.set_title(f'Feature Patterns (B{batch_idx})')
            ax2.set_xlabel('Feature Dimension')
            ax2.set_ylabel('Slot Index')
            
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Normalized Feature Value')
            
            # Alpha value distributions
            ax3 = fig.add_subplot(gs[i, 2])
            
            if 'spatial' in alphas:
                spatial_alpha = alphas['spatial'][batch_idx, :, 0].cpu().numpy()
                ax3.bar(range(n_slots), spatial_alpha, alpha=0.7, 
                       color='blue', label='Spatial Alpha')
            
            if 'feature' in alphas:
                feature_alpha = alphas['feature'][batch_idx, :, 0].cpu().numpy()
                ax3.bar(range(n_slots), feature_alpha, alpha=0.7, 
                       color='red', label='Feature Alpha', width=0.6)
            
            ax3.set_title(f'Alpha Values (B{batch_idx})')
            ax3.set_xlabel('Slot Index')
            ax3.set_ylabel('Alpha Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Feature diversity analysis
            ax4 = fig.add_subplot(gs[i, 3])
            
            # Compute feature statistics per slot
            feature_means = np.mean(features, axis=1)
            feature_stds = np.std(features, axis=1)
            feature_maxs = np.max(features, axis=1)
            
            x_pos = np.arange(n_slots)
            width = 0.25
            
            ax4.bar(x_pos - width, feature_means, width, label='Mean', alpha=0.8)
            ax4.bar(x_pos, feature_stds, width, label='Std', alpha=0.8)
            ax4.bar(x_pos + width, feature_maxs, width, label='Max', alpha=0.8)
            
            ax4.set_title(f'Feature Statistics (B{batch_idx})')
            ax4.set_xlabel('Slot Index')
            ax4.set_ylabel('Feature Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_reconstruction_quality(
        self,
        original_images: torch.Tensor,
        reconstructions: torch.Tensor,
        slot_reconstructions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        batch_indices: Optional[List[int]] = None,
        compute_metrics: bool = True
    ) -> plt.Figure:
        """
        Visualize reconstruction quality per slot and overall.
        
        Args:
            original_images: Original input images (B, C, H, W)
            reconstructions: Overall reconstructed images (B, C, H, W)
            slot_reconstructions: Per-slot reconstructions (B, n_slots, C, H, W)
            masks: Object masks (B, n_slots, H, W)
            save_path: Optional path to save the figure
            batch_indices: Which batch samples to visualize
            compute_metrics: Whether to compute and display quality metrics
            
        Returns:
            Matplotlib figure object
        """
        B, C, H, W = original_images.shape
        
        if batch_indices is None:
            batch_indices = list(range(min(3, B)))
        
        n_batch = len(batch_indices)
        n_cols = 3  # Original, reconstruction, difference
        
        if slot_reconstructions is not None:
            n_slots = slot_reconstructions.shape[1]
            n_cols += n_slots
        
        # Create figure
        fig, axes = plt.subplots(
            n_batch, n_cols,
            figsize=(n_cols * 2.5, n_batch * 2.5),
            squeeze=False
        )
        
        for i, batch_idx in enumerate(batch_indices):
            orig = original_images[batch_idx].cpu()
            recon = reconstructions[batch_idx].cpu()
            
            # Convert images for display
            if C == 1:
                orig_np = orig.squeeze(0).numpy()
                recon_np = recon.squeeze(0).numpy()
                cmap_img = 'gray'
            else:
                orig_np = orig.permute(1, 2, 0).numpy()
                recon_np = recon.permute(1, 2, 0).numpy()
                orig_np = np.clip(orig_np, 0, 1)
                recon_np = np.clip(recon_np, 0, 1)
                cmap_img = None
            
            col_idx = 0
            
            # Original image
            axes[i, col_idx].imshow(orig_np, cmap=cmap_img)
            axes[i, col_idx].set_title(f'Original (B{batch_idx})')
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Reconstruction
            axes[i, col_idx].imshow(recon_np, cmap=cmap_img)
            
            if compute_metrics:
                # Compute MSE
                mse = F.mse_loss(recon, orig).item()
                axes[i, col_idx].set_title(f'Reconstruction\nMSE: {mse:.4f}')
            else:
                axes[i, col_idx].set_title('Reconstruction')
            
            axes[i, col_idx].axis('off')
            col_idx += 1
            
            # Difference map
            if C == 1:
                diff = np.abs(orig_np - recon_np)
            else:
                diff = np.mean(np.abs(orig_np - recon_np), axis=2)
            
            im = axes[i, col_idx].imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
            axes[i, col_idx].set_title('Absolute Difference')
            axes[i, col_idx].axis('off')
            
            if i == 0:
                cbar = plt.colorbar(im, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
                cbar.set_label('|Original - Reconstruction|')
            col_idx += 1
            
            # Individual slot reconstructions
            if slot_reconstructions is not None:
                for slot_idx in range(n_slots):
                    slot_recon = slot_reconstructions[batch_idx, slot_idx].cpu()
                    
                    if C == 1:
                        slot_recon_np = slot_recon.squeeze(0).numpy()
                    else:
                        slot_recon_np = slot_recon.permute(1, 2, 0).numpy()
                        slot_recon_np = np.clip(slot_recon_np, 0, 1)
                    
                    axes[i, col_idx].imshow(slot_recon_np, cmap=cmap_img)
                    
                    # Apply mask if available
                    if masks is not None:
                        mask = masks[batch_idx, slot_idx].cpu().numpy()
                        # Create masked version with transparency
                        masked_recon = slot_recon_np.copy()
                        if C == 3:
                            mask_rgb = np.stack([mask, mask, mask], axis=2)
                            masked_recon = masked_recon * mask_rgb
                        else:
                            masked_recon = masked_recon * mask
                        axes[i, col_idx].imshow(masked_recon, cmap=cmap_img)
                    
                    axes[i, col_idx].set_title(f'Slot {slot_idx}')
                    axes[i, col_idx].axis('off')
                    col_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===============================
    # 2. TRAINING PROGRESS VISUALIZATION
    # ===============================
    
    def plot_training_curves(
        self,
        training_logs: Dict[str, List[float]],
        validation_logs: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None,
        metrics_to_plot: Optional[List[str]] = None,
        log_scale: bool = False,
        smoothing: float = 0.0
    ) -> plt.Figure:
        """
        Plot training and validation curves for various metrics.
        
        Args:
            training_logs: Dictionary with metric names and values over epochs
            validation_logs: Optional validation logs
            save_path: Optional path to save the figure
            metrics_to_plot: Which metrics to plot (defaults to all)
            log_scale: Whether to use log scale for y-axis
            smoothing: Exponential smoothing factor (0 = no smoothing)
            
        Returns:
            Matplotlib figure object
        """
        if metrics_to_plot is None:
            metrics_to_plot = list(training_logs.keys())
        
        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            # Handle empty metrics case
            fig, ax = plt.subplots(figsize=self.config.figsize)
            ax.text(0.5, 0.5, 'No metrics to plot', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('Training Curves')
            return fig
        
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5, n_rows * 4),
            squeeze=False
        )
        
        for i, metric in enumerate(metrics_to_plot):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            if metric not in training_logs:
                ax.text(0.5, 0.5, f'Metric "{metric}" not found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric)
                continue
            
            train_values = training_logs[metric]
            epochs = list(range(len(train_values)))
            
            # Apply smoothing if requested
            if smoothing > 0:
                train_values = self._apply_exponential_smoothing(train_values, smoothing)
            
            # Plot training curve
            ax.plot(epochs, train_values, label='Training', 
                   color='blue', linewidth=self.config.line_width)
            
            # Plot validation curve if available
            if validation_logs and metric in validation_logs:
                val_values = validation_logs[metric]
                val_epochs = list(range(len(val_values)))
                
                if smoothing > 0:
                    val_values = self._apply_exponential_smoothing(val_values, smoothing)
                
                ax.plot(val_epochs, val_values, label='Validation', 
                       color='red', linewidth=self.config.line_width)
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=self.config.grid_alpha)
            
            if log_scale:
                ax.set_yscale('log')
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_curriculum_progression(
        self,
        curriculum_logs: Dict[str, List[float]],
        difficulty_schedule: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize curriculum learning progression.
        
        Args:
            curriculum_logs: Logs related to curriculum learning
            difficulty_schedule: Optional difficulty schedule over epochs
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Difficulty progression
        if difficulty_schedule is not None:
            epochs = list(range(len(difficulty_schedule)))
            axes[0, 0].plot(epochs, difficulty_schedule, 
                          color='purple', linewidth=self.config.line_width)
            axes[0, 0].set_title('Curriculum Difficulty Schedule')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Difficulty Level')
            axes[0, 0].grid(True, alpha=self.config.grid_alpha)
        
        # Plot 2: Performance by difficulty
        if 'performance_by_difficulty' in curriculum_logs:
            perf_data = curriculum_logs['performance_by_difficulty']
            for difficulty, perfs in perf_data.items():
                axes[0, 1].plot(perfs, label=f'Difficulty {difficulty}', 
                              linewidth=self.config.line_width)
            axes[0, 1].set_title('Performance by Difficulty Level')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Performance')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=self.config.grid_alpha)
        
        # Plot 3: Sample complexity over time
        if 'samples_per_difficulty' in curriculum_logs:
            sample_data = curriculum_logs['samples_per_difficulty']
            difficulties = list(sample_data.keys())
            sample_counts = [sum(counts) for counts in sample_data.values()]
            
            axes[1, 0].bar(difficulties, sample_counts, alpha=0.7)
            axes[1, 0].set_title('Total Samples by Difficulty')
            axes[1, 0].set_xlabel('Difficulty Level')
            axes[1, 0].set_ylabel('Sample Count')
            axes[1, 0].grid(True, alpha=self.config.grid_alpha)
        
        # Plot 4: Success rate progression
        if 'success_rate' in curriculum_logs:
            success_rates = curriculum_logs['success_rate']
            epochs = list(range(len(success_rates)))
            axes[1, 1].plot(epochs, success_rates, 
                          color='green', linewidth=self.config.line_width)
            axes[1, 1].set_title('Overall Success Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_alpha_evolution(
        self,
        alpha_logs: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None,
        show_distribution: bool = True
    ) -> plt.Figure:
        """
        Visualize evolution of alpha values during training.
        
        Args:
            alpha_logs: Nested dict with alpha type -> statistic -> values
            save_path: Optional path to save the figure
            show_distribution: Whether to show distribution plots
            
        Returns:
            Matplotlib figure object
        """
        alpha_types = list(alpha_logs.keys())
        n_types = len(alpha_types)
        
        if show_distribution:
            fig, axes = plt.subplots(n_types, 3, figsize=(15, 5 * n_types))
            if n_types == 1:
                axes = axes.reshape(1, -1)
        else:
            fig, axes = plt.subplots(n_types, 2, figsize=(12, 4 * n_types))
            if n_types == 1:
                axes = axes.reshape(1, -1)
        
        for i, alpha_type in enumerate(alpha_types):
            type_logs = alpha_logs[alpha_type]
            
            # Plot mean and std evolution
            ax1 = axes[i, 0]
            if 'mean' in type_logs:
                epochs = list(range(len(type_logs['mean'])))
                ax1.plot(epochs, type_logs['mean'], label='Mean', 
                        color='blue', linewidth=self.config.line_width)
                
                if 'std' in type_logs:
                    mean_vals = np.array(type_logs['mean'])
                    std_vals = np.array(type_logs['std'])
                    ax1.fill_between(epochs, 
                                   mean_vals - std_vals, 
                                   mean_vals + std_vals,
                                   alpha=0.3, color='blue', label='±1 Std')
            
            ax1.set_title(f'{alpha_type.title()} Alpha Evolution')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Alpha Value')
            ax1.legend()
            ax1.grid(True, alpha=self.config.grid_alpha)
            
            # Plot min/max range
            ax2 = axes[i, 1]
            if 'min' in type_logs and 'max' in type_logs:
                epochs = list(range(len(type_logs['min'])))
                ax2.plot(epochs, type_logs['min'], label='Min', 
                        color='red', linewidth=self.config.line_width)
                ax2.plot(epochs, type_logs['max'], label='Max', 
                        color='green', linewidth=self.config.line_width)
                ax2.fill_between(epochs, type_logs['min'], type_logs['max'],
                               alpha=0.3, color='gray', label='Range')
            
            ax2.set_title(f'{alpha_type.title()} Alpha Range')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Alpha Value')
            ax2.legend()
            ax2.grid(True, alpha=self.config.grid_alpha)
            
            # Plot distribution over time (if requested)
            if show_distribution and 'distribution' in type_logs:
                ax3 = axes[i, 2]
                dist_data = type_logs['distribution']
                
                # Create heatmap of alpha distributions over time
                if isinstance(dist_data, list) and len(dist_data) > 0:
                    # Assume dist_data is list of histograms over epochs
                    hist_matrix = np.array(dist_data).T
                    im = ax3.imshow(hist_matrix, aspect='auto', cmap=self.cmap_alpha,
                                  origin='lower', interpolation='nearest')
                    ax3.set_title(f'{alpha_type.title()} Alpha Distribution')
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Alpha Value Bins')
                    
                    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                    cbar.set_label('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_binding_consistency_trends(
        self,
        consistency_logs: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize binding consistency trends over training.
        
        Args:
            consistency_logs: Logs of various consistency metrics
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        metrics = list(consistency_logs.keys())
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 6, n_rows * 4),
            squeeze=False
        )
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_metrics))
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            values = consistency_logs[metric]
            epochs = list(range(len(values)))
            
            ax.plot(epochs, values, color=colors[i], 
                   linewidth=self.config.line_width, label=metric)
            
            # Add trend line
            if len(values) > 1:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "--", color=colors[i], alpha=0.7)
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Consistency Score')
            ax.grid(True, alpha=self.config.grid_alpha)
            
            # Add statistics text
            if len(values) > 0:
                final_value = values[-1]
                mean_value = np.mean(values)
                ax.text(0.02, 0.98, f'Final: {final_value:.3f}\nMean: {mean_value:.3f}',
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===============================
    # 3. COMPARATIVE ANALYSIS
    # ===============================
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics_to_compare: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        plot_type: str = 'bar'  # 'bar', 'radar', 'line'
    ) -> plt.Figure:
        """
        Compare different models side-by-side.
        
        Args:
            model_results: Dict of model_name -> metric_name -> value
            metrics_to_compare: Which metrics to include in comparison
            save_path: Optional path to save the figure
            plot_type: Type of comparison plot
            
        Returns:
            Matplotlib figure object
        """
        model_names = list(model_results.keys())
        
        if metrics_to_compare is None:
            # Get all metrics that appear in all models
            all_metrics = set()
            for model_metrics in model_results.values():
                all_metrics.update(model_metrics.keys())
            
            # Keep only metrics present in all models
            metrics_to_compare = []
            for metric in all_metrics:
                if all(metric in model_results[model] for model in model_names):
                    metrics_to_compare.append(metric)
        
        if plot_type == 'bar':
            return self._plot_bar_comparison(model_results, metrics_to_compare, save_path)
        elif plot_type == 'radar':
            return self._plot_radar_comparison(model_results, metrics_to_compare, save_path)
        elif plot_type == 'line':
            return self._plot_line_comparison(model_results, metrics_to_compare, save_path)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    def plot_ablation_study(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        baseline_name: str = 'full_model',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize ablation study results.
        
        Args:
            ablation_results: Dict of ablation_name -> metric_name -> value
            baseline_name: Name of the baseline (full) model
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if baseline_name not in ablation_results:
            raise ValueError(f"Baseline '{baseline_name}' not found in results")
        
        baseline_results = ablation_results[baseline_name]
        metrics = list(baseline_results.keys())
        
        # Calculate relative performance
        relative_results = {}
        for ablation_name, results in ablation_results.items():
            if ablation_name != baseline_name:
                relative_results[ablation_name] = {}
                for metric in metrics:
                    if metric in results:
                        baseline_val = baseline_results[metric]
                        if baseline_val != 0:
                            relative_results[ablation_name][metric] = \
                                (results[metric] - baseline_val) / abs(baseline_val) * 100
                        else:
                            relative_results[ablation_name][metric] = 0
        
        # Create visualization
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Absolute values
        ax1 = axes[0]
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(ablation_results)
        
        for i, (model_name, results) in enumerate(ablation_results.items()):
            values = [results.get(metric, 0) for metric in metrics]
            offset = (i - len(ablation_results) / 2 + 0.5) * width
            
            color = 'red' if model_name == baseline_name else None
            ax1.bar(x_pos + offset, values, width, label=model_name, 
                   color=color, alpha=0.8)
        
        ax1.set_title('Absolute Performance')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Value')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Plot 2: Relative performance (percentage change from baseline)
        ax2 = axes[1]
        
        ablation_names = list(relative_results.keys())
        if ablation_names:
            x_pos = np.arange(len(metrics))
            width = 0.8 / len(ablation_names)
            
            for i, (ablation_name, results) in enumerate(relative_results.items()):
                values = [results.get(metric, 0) for metric in metrics]
                offset = (i - len(ablation_names) / 2 + 0.5) * width
                
                colors = ['red' if v < 0 else 'green' for v in values]
                ax2.bar(x_pos + offset, values, width, label=ablation_name, 
                       color=colors, alpha=0.7)
        
        ax2.set_title('Relative Performance (% change from baseline)')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Percentage Change (%)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_before_after_training(
        self,
        before_results: Dict[str, torch.Tensor],
        after_results: Dict[str, torch.Tensor],
        sample_images: torch.Tensor,
        save_path: Optional[str] = None,
        sample_indices: Optional[List[int]] = None
    ) -> plt.Figure:
        """
        Compare model outputs before and after training.
        
        Args:
            before_results: Model outputs before training
            after_results: Model outputs after training
            sample_images: Input images for comparison
            save_path: Optional path to save the figure
            sample_indices: Which samples to show
            
        Returns:
            Matplotlib figure object
        """
        B, C, H, W = sample_images.shape
        
        if sample_indices is None:
            sample_indices = list(range(min(3, B)))
        
        n_samples = len(sample_indices)
        
        # Create figure with grid layout
        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_idx in enumerate(sample_indices):
            # Original image
            img = sample_images[sample_idx].cpu()
            if C == 1:
                img_np = img.squeeze(0).numpy()
                cmap_img = 'gray'
            else:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                cmap_img = None
            
            axes[i, 0].imshow(img_np, cmap=cmap_img)
            axes[i, 0].set_title(f'Original (Sample {sample_idx})')
            axes[i, 0].axis('off')
            
            # Before training - reconstruction
            before_recon = before_results['reconstruction'][sample_idx].cpu()
            if C == 1:
                before_recon_np = before_recon.squeeze(0).numpy()
            else:
                before_recon_np = before_recon.permute(1, 2, 0).numpy()
                before_recon_np = np.clip(before_recon_np, 0, 1)
            
            axes[i, 1].imshow(before_recon_np, cmap=cmap_img)
            axes[i, 1].set_title('Before Training')
            axes[i, 1].axis('off')
            
            # Before training - attention
            before_attention = before_results['spatial_attention'][sample_idx].sum(0).cpu().numpy()
            axes[i, 2].imshow(before_attention, cmap=self.cmap_attention)
            axes[i, 2].set_title('Before - Attention')
            axes[i, 2].axis('off')
            
            # After training - reconstruction
            after_recon = after_results['reconstruction'][sample_idx].cpu()
            if C == 1:
                after_recon_np = after_recon.squeeze(0).numpy()
            else:
                after_recon_np = after_recon.permute(1, 2, 0).numpy()
                after_recon_np = np.clip(after_recon_np, 0, 1)
            
            axes[i, 3].imshow(after_recon_np, cmap=cmap_img)
            axes[i, 3].set_title('After Training')
            axes[i, 3].axis('off')
            
            # After training - attention
            after_attention = after_results['spatial_attention'][sample_idx].sum(0).cpu().numpy()
            axes[i, 4].imshow(after_attention, cmap=self.cmap_attention)
            axes[i, 4].set_title('After - Attention')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===============================
    # 4. INTERACTIVE ANALYSIS
    # ===============================
    
    def create_interactive_slot_explorer(
        self,
        model_output: Dict[str, torch.Tensor],
        images: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Create interactive widget for exploring slot assignments.
        
        Args:
            model_output: MVBA model output dictionary
            images: Input images
            save_path: Optional path to save widget state
        """
        if not JUPYTER_AVAILABLE:
            logger.warning("Jupyter widgets not available, creating static plots instead")
            return self.plot_slot_assignments(images, model_output['masks'], 
                                            model_output['spatial_attention'])
        
        B, C, H, W = images.shape
        n_slots = model_output['masks'].shape[1]
        
        def plot_sample_slot(batch_idx=0, slot_idx=0, overlay_alpha=0.6):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original image
            img = images[batch_idx].cpu()
            if C == 1:
                img_np = img.squeeze(0).numpy()
                cmap_img = 'gray'
            else:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                cmap_img = None
            
            axes[0].imshow(img_np, cmap=cmap_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Slot mask
            mask = model_output['masks'][batch_idx, slot_idx].cpu().numpy()
            axes[1].imshow(mask, cmap='hot')
            axes[1].set_title(f'Slot {slot_idx} Mask')
            axes[1].axis('off')
            
            # Spatial attention
            attention = model_output['spatial_attention'][batch_idx, slot_idx].cpu().numpy()
            axes[2].imshow(img_np, cmap=cmap_img)
            axes[2].imshow(attention, alpha=overlay_alpha, cmap=self.cmap_attention)
            axes[2].set_title(f'Slot {slot_idx} Attention')
            axes[2].axis('off')
            
            # Feature visualization (simplified)
            if 'bound_features' in model_output:
                features = model_output['bound_features'][batch_idx, slot_idx].cpu().numpy()
                feature_norm = np.linalg.norm(features)
                axes[3].bar(range(min(20, len(features))), features[:20])
                axes[3].set_title(f'Features (norm: {feature_norm:.2f})')
                axes[3].set_xlabel('Feature Dimension')
            
            plt.tight_layout()
            plt.show()
        
        # Create interactive widget
        batch_slider = IntSlider(min=0, max=B-1, step=1, value=0, description='Batch:')
        slot_slider = IntSlider(min=0, max=n_slots-1, step=1, value=0, description='Slot:')
        alpha_slider = FloatSlider(min=0.0, max=1.0, step=0.1, value=0.6, description='Alpha:')
        
        interactive_plot = interactive(plot_sample_slot, 
                                     batch_idx=batch_slider,
                                     slot_idx=slot_slider,
                                     overlay_alpha=alpha_slider)
        
        display(interactive_plot)
        
        # Save widget state if requested
        if save_path:
            widget_state = {
                'batch_idx': batch_slider.value,
                'slot_idx': slot_slider.value,
                'overlay_alpha': alpha_slider.value
            }
            with open(save_path, 'w') as f:
                json.dump(widget_state, f)
    
    def create_interactive_training_dashboard(
        self,
        training_logs: Dict[str, List[float]],
        validation_logs: Optional[Dict[str, List[float]]] = None
    ):
        """
        Create interactive dashboard for training progress.
        
        Args:
            training_logs: Training metrics over time
            validation_logs: Optional validation metrics
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, creating static plots instead")
            return self.plot_training_curves(training_logs, validation_logs)
        
        # Create plotly dashboard
        metrics = list(training_logs.keys())
        n_metrics = len(metrics)
        
        # Create subplots
        fig = make_subplots(
            rows=(n_metrics + 1) // 2,
            cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics]
        )
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            epochs = list(range(len(training_logs[metric])))
            
            # Add training curve
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=training_logs[metric],
                    mode='lines',
                    name=f'{metric} (train)',
                    line=dict(color='blue', width=2)
                ),
                row=row, col=col
            )
            
            # Add validation curve if available
            if validation_logs and metric in validation_logs:
                val_epochs = list(range(len(validation_logs[metric])))
                fig.add_trace(
                    go.Scatter(
                        x=val_epochs,
                        y=validation_logs[metric],
                        mode='lines',
                        name=f'{metric} (val)',
                        line=dict(color='red', width=2)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Interactive Training Dashboard',
            height=300 * ((n_metrics + 1) // 2),
            showlegend=True
        )
        
        fig.show()
        
        return fig
    
    def save_visualization_state(
        self,
        state_dict: Dict[str, Any],
        save_path: str
    ):
        """
        Save current visualization state for later loading.
        
        Args:
            state_dict: Dictionary containing visualization state
            save_path: Path to save the state
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy for serialization
        serializable_state = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                serializable_state[key] = value.cpu().numpy()
            elif isinstance(value, dict):
                serializable_state[key] = {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                serializable_state[key] = value
        
        with open(save_path, 'wb') as f:
            pickle.dump(serializable_state, f)
        
        logger.info(f"Visualization state saved to {save_path}")
    
    def load_visualization_state(
        self,
        load_path: str
    ) -> Dict[str, Any]:
        """
        Load previously saved visualization state.
        
        Args:
            load_path: Path to load the state from
            
        Returns:
            Dictionary containing visualization state
        """
        with open(load_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        logger.info(f"Visualization state loaded from {load_path}")
        return state_dict
    
    # ===============================
    # 5. DIAGNOSTIC TOOLS
    # ===============================
    
    def analyze_attention_patterns(
        self,
        spatial_attention: torch.Tensor,
        save_path: Optional[str] = None,
        batch_indices: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Analyze and visualize attention patterns for diagnostic purposes.
        
        Args:
            spatial_attention: Spatial attention maps (B, n_slots, H, W)
            save_path: Optional path to save analysis
            batch_indices: Which batches to analyze
            
        Returns:
            Dictionary with attention pattern statistics
        """
        B, n_slots, H, W = spatial_attention.shape
        
        if batch_indices is None:
            batch_indices = list(range(B))
        
        # Compute attention statistics
        stats = {}
        
        # 1. Entropy analysis
        entropies = []
        for b in batch_indices:
            for s in range(n_slots):
                attention = spatial_attention[b, s].flatten()
                attention_norm = attention / (attention.sum() + 1e-8)
                entropy = -(attention_norm * torch.log(attention_norm + 1e-8)).sum()
                entropies.append(entropy.item())
        
        stats['mean_entropy'] = np.mean(entropies)
        stats['std_entropy'] = np.std(entropies)
        
        # 2. Sparsity analysis
        sparsities = []
        for b in batch_indices:
            for s in range(n_slots):
                attention = spatial_attention[b, s]
                sparsity = (attention > 0.1 * attention.max()).float().mean()
                sparsities.append(sparsity.item())
        
        stats['mean_sparsity'] = np.mean(sparsities)
        stats['std_sparsity'] = np.std(sparsities)
        
        # 3. Overlap analysis
        overlaps = []
        for b in batch_indices:
            for i in range(n_slots):
                for j in range(i + 1, n_slots):
                    att_i = spatial_attention[b, i].flatten()
                    att_j = spatial_attention[b, j].flatten()
                    
                    att_i_norm = att_i / (att_i.sum() + 1e-8)
                    att_j_norm = att_j / (att_j.sum() + 1e-8)
                    
                    overlap = (torch.minimum(att_i_norm, att_j_norm)).sum()
                    overlaps.append(overlap.item())
        
        stats['mean_overlap'] = np.mean(overlaps) if overlaps else 0.0
        stats['std_overlap'] = np.std(overlaps) if overlaps else 0.0
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot entropy distribution
        axes[0, 0].hist(entropies, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Attention Entropy Distribution')
        axes[0, 0].set_xlabel('Entropy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(stats['mean_entropy'], color='red', linestyle='--', 
                          label=f'Mean: {stats["mean_entropy"]:.2f}')
        axes[0, 0].legend()
        
        # Plot sparsity distribution
        axes[0, 1].hist(sparsities, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Attention Sparsity Distribution')
        axes[0, 1].set_xlabel('Sparsity (fraction > 0.1*max)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(stats['mean_sparsity'], color='red', linestyle='--',
                          label=f'Mean: {stats["mean_sparsity"]:.2f}')
        axes[0, 1].legend()
        
        # Plot overlap distribution
        if overlaps:
            axes[1, 0].hist(overlaps, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Slot Overlap Distribution')
            axes[1, 0].set_xlabel('Overlap')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(stats['mean_overlap'], color='red', linestyle='--',
                              label=f'Mean: {stats["mean_overlap"]:.2f}')
            axes[1, 0].legend()
        
        # Plot attention heatmap for one sample
        if batch_indices:
            sample_idx = batch_indices[0]
            combined_attention = spatial_attention[sample_idx].sum(0).cpu().numpy()
            im = axes[1, 1].imshow(combined_attention, cmap=self.cmap_attention)
            axes[1, 1].set_title(f'Combined Attention (Sample {sample_idx})')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return stats
    
    def plot_slot_utilization_over_time(
        self,
        slot_usage_logs: List[torch.Tensor],
        save_path: Optional[str] = None,
        epoch_interval: int = 1
    ) -> plt.Figure:
        """
        Visualize how slot utilization changes over training epochs.
        
        Args:
            slot_usage_logs: List of slot usage tensors over epochs
            save_path: Optional path to save the figure
            epoch_interval: Interval between logged epochs
            
        Returns:
            Matplotlib figure object
        """
        n_epochs = len(slot_usage_logs)
        n_slots = slot_usage_logs[0].shape[1] if slot_usage_logs else 0
        
        if n_epochs == 0:
            logger.warning("No slot usage data provided")
            return plt.figure()
        
        # Calculate utilization over time
        utilization_over_time = np.zeros((n_epochs, n_slots))
        
        for epoch, usage_tensor in enumerate(slot_usage_logs):
            # Average usage across batch and spatial dimensions
            avg_usage = usage_tensor.mean(dim=(0, 2, 3)).cpu().numpy()  # Average over batch, height, width
            utilization_over_time[epoch] = avg_usage
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Utilization heatmap over time
        epochs = np.arange(n_epochs) * epoch_interval
        im1 = axes[0, 0].imshow(utilization_over_time.T, aspect='auto', 
                               cmap='viridis', origin='lower')
        axes[0, 0].set_title('Slot Utilization Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Slot Index')
        
        # Set x-tick labels to actual epochs
        x_ticks = np.linspace(0, n_epochs-1, min(10, n_epochs)).astype(int)
        axes[0, 0].set_xticks(x_ticks)
        axes[0, 0].set_xticklabels([epochs[i] for i in x_ticks])
        
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        cbar1.set_label('Utilization')
        
        # Plot 2: Per-slot utilization trends
        for slot_idx in range(n_slots):
            axes[0, 1].plot(epochs, utilization_over_time[:, slot_idx], 
                           label=f'Slot {slot_idx}', linewidth=1.5)
        
        axes[0, 1].set_title('Individual Slot Utilization Trends')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Utilization')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Average utilization over time
        avg_utilization = utilization_over_time.mean(axis=1)
        std_utilization = utilization_over_time.std(axis=1)
        
        axes[1, 0].plot(epochs, avg_utilization, color='blue', linewidth=2)
        axes[1, 0].fill_between(epochs, 
                               avg_utilization - std_utilization,
                               avg_utilization + std_utilization,
                               alpha=0.3, color='blue')
        axes[1, 0].set_title('Average Slot Utilization')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Average Utilization')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Utilization variance (load balancing)
        utilization_var = utilization_over_time.var(axis=1)
        
        axes[1, 1].plot(epochs, utilization_var, color='red', linewidth=2)
        axes[1, 1].set_title('Slot Load Balancing (Lower = Better)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Utilization Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def visualize_gradient_flow(
        self,
        model: torch.nn.Module,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize gradient flow through the model for debugging.
        
        Args:
            model: MVBA model with computed gradients
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Collect gradient statistics
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                grad_mean = param.grad.abs().mean().item()
                
                grad_stats[name] = {
                    'norm': grad_norm,
                    'max': grad_max,
                    'mean': grad_mean
                }
        
        if not grad_stats:
            logger.warning("No gradients found in model")
            return plt.figure()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        param_names = list(grad_stats.keys())
        grad_norms = [grad_stats[name]['norm'] for name in param_names]
        grad_maxs = [grad_stats[name]['max'] for name in param_names]
        grad_means = [grad_stats[name]['mean'] for name in param_names]
        
        # Plot 1: Gradient norms
        axes[0, 0].bar(range(len(param_names)), grad_norms, alpha=0.7)
        axes[0, 0].set_title('Gradient Norms by Parameter')
        axes[0, 0].set_xlabel('Parameter Index')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Gradient max values
        axes[0, 1].bar(range(len(param_names)), grad_maxs, alpha=0.7, color='orange')
        axes[0, 1].set_title('Maximum Gradient Values')
        axes[0, 1].set_xlabel('Parameter Index')
        axes[0, 1].set_ylabel('Max Gradient')
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Gradient mean values
        axes[1, 0].bar(range(len(param_names)), grad_means, alpha=0.7, color='green')
        axes[1, 0].set_title('Mean Gradient Values')
        axes[1, 0].set_xlabel('Parameter Index')
        axes[1, 0].set_ylabel('Mean Gradient')
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Gradient distribution (histogram)
        all_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.flatten().cpu().numpy())
        
        if all_grads:
            axes[1, 1].hist(all_grads, bins=50, alpha=0.7, color='purple')
            axes[1, 1].set_title('Overall Gradient Distribution')
            axes[1, 1].set_xlabel('Gradient Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_yscale('log')
        
        # Add parameter names as x-tick labels (rotated)
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.set_xticks(range(len(param_names)))
            ax.set_xticklabels([name.split('.')[-1] for name in param_names], 
                              rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def analyze_failure_modes(
        self,
        model_outputs: List[Dict[str, torch.Tensor]],
        target_images: List[torch.Tensor],
        failure_threshold: float = 0.1,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze and visualize common failure modes.
        
        Args:
            model_outputs: List of model outputs from different samples
            target_images: List of corresponding target images
            failure_threshold: Threshold for considering a sample as failure
            save_path: Optional path to save analysis
            
        Returns:
            Dictionary with failure mode analysis
        """
        failures = []
        successes = []
        
        # Identify failures based on reconstruction error
        for i, (output, target) in enumerate(zip(model_outputs, target_images)):
            recon = output['reconstruction']
            mse = F.mse_loss(recon, target).item()
            
            if mse > failure_threshold:
                failures.append({
                    'index': i,
                    'mse': mse,
                    'output': output,
                    'target': target
                })
            else:
                successes.append({
                    'index': i,
                    'mse': mse,
                    'output': output,
                    'target': target
                })
        
        # Analyze failure patterns
        analysis = {
            'n_failures': len(failures),
            'n_successes': len(successes),
            'failure_rate': len(failures) / len(model_outputs),
            'mean_failure_mse': np.mean([f['mse'] for f in failures]) if failures else 0,
            'mean_success_mse': np.mean([s['mse'] for s in successes]) if successes else 0
        }
        
        # Create visualization
        if failures:
            n_show = min(4, len(failures))
            fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
            if n_show == 1:
                axes = axes.reshape(1, -1)
            
            for i, failure in enumerate(failures[:n_show]):
                output = failure['output']
                target = failure['target']
                
                # Original
                img = target.cpu()
                if img.dim() == 4:
                    img = img[0]  # Remove batch dimension if present
                
                if img.shape[0] == 1:
                    img_np = img.squeeze(0).numpy()
                    cmap_img = 'gray'
                else:
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = np.clip(img_np, 0, 1)
                    cmap_img = None
                
                axes[i, 0].imshow(img_np, cmap=cmap_img)
                axes[i, 0].set_title(f'Target (MSE: {failure["mse"]:.4f})')
                axes[i, 0].axis('off')
                
                # Reconstruction
                recon = output['reconstruction'].cpu()
                if recon.dim() == 4:
                    recon = recon[0]
                
                if recon.shape[0] == 1:
                    recon_np = recon.squeeze(0).numpy()
                else:
                    recon_np = recon.permute(1, 2, 0).numpy()
                    recon_np = np.clip(recon_np, 0, 1)
                
                axes[i, 1].imshow(recon_np, cmap=cmap_img)
                axes[i, 1].set_title('Reconstruction')
                axes[i, 1].axis('off')
                
                # Attention
                if 'spatial_attention' in output:
                    attention = output['spatial_attention']
                    if attention.dim() == 4:
                        attention = attention[0]  # Remove batch dimension
                    
                    combined_attention = attention.sum(0).cpu().numpy()
                    axes[i, 2].imshow(combined_attention, cmap=self.cmap_attention)
                    axes[i, 2].set_title('Spatial Attention')
                    axes[i, 2].axis('off')
                
                # Masks
                if 'masks' in output:
                    masks = output['masks']
                    if masks.dim() == 4:
                        masks = masks[0]
                    
                    colored_masks = self._create_colored_segmentation(masks)
                    axes[i, 3].imshow(colored_masks)
                    axes[i, 3].set_title('Slot Assignments')
                    axes[i, 3].axis('off')
            
            plt.suptitle(f'Failure Mode Analysis (Showing {n_show}/{len(failures)} failures)')
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
        else:
            fig = plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No failures detected!', 
                    ha='center', va='center', fontsize=20, 
                    transform=plt.gca().transAxes)
            plt.title('Failure Mode Analysis')
        
        return analysis
    
    # ===============================
    # HELPER METHODS
    # ===============================
    
    def _create_colored_segmentation(self, masks: torch.Tensor) -> np.ndarray:
        """Create colored segmentation from slot masks."""
        n_slots, H, W = masks.shape
        colored = np.zeros((H, W, 3))
        
        # Assign each pixel to the slot with highest probability
        assignments = masks.argmax(dim=0).cpu().numpy()
        
        for slot_idx in range(n_slots):
            color = self.colors_slots[slot_idx % len(self.colors_slots)]
            # Convert hex color to RGB
            if isinstance(color, str) and color.startswith('#'):
                color = [int(color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
            
            mask = (assignments == slot_idx)
            colored[mask] = color
        
        return colored
    
    def _apply_exponential_smoothing(self, values: List[float], alpha: float) -> List[float]:
        """Apply exponential smoothing to a list of values."""
        if not values:
            return values
        
        smoothed = [values[0]]
        for value in values[1:]:
            smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
        
        return smoothed
    
    def _plot_bar_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create bar chart comparison between models."""
        model_names = list(model_results.keys())
        n_metrics = len(metrics)
        n_models = len(model_names)
        
        fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 6))
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        for i, model_name in enumerate(model_names):
            values = [model_results[model_name].get(metric, 0) for metric in metrics]
            offset = (i - n_models / 2 + 0.5) * width
            
            ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        ax.set_title('Model Performance Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_radar_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create radar chart comparison between models."""
        model_names = list(model_results.keys())
        n_metrics = len(metrics)
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize values to [0, 1] for radar chart
        all_values = []
        for model_name in model_names:
            values = [model_results[model_name].get(metric, 0) for metric in metrics]
            all_values.extend(values)
        
        min_val, max_val = min(all_values), max(all_values)
        value_range = max_val - min_val if max_val != min_val else 1
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            values = [model_results[model_name].get(metric, 0) for metric in metrics]
            # Normalize values
            norm_values = [(v - min_val) / value_range for v in values]
            norm_values += norm_values[:1]  # Complete the circle
            
            ax.plot(angles, norm_values, 'o-', linewidth=2, 
                   label=model_name, color=colors[i])
            ax.fill(angles, norm_values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_line_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create line plot comparison between models."""
        model_names = list(model_results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            values = [model_results[model_name].get(metric, 0) for metric in metrics]
            ax.plot(range(len(metrics)), values, 'o-', 
                   linewidth=2, markersize=8, label=model_name, color=colors[i])
        
        ax.set_title('Model Performance Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, save_path: str):
        """Save figure with proper configuration."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension or use config default
        if save_path.suffix:
            fmt = save_path.suffix[1:]  # Remove the dot
        else:
            fmt = self.config.save_format
            save_path = save_path.with_suffix(f'.{fmt}')
        
        # Set DPI based on quality setting
        dpi = self.config.dpi * 2 if self.config.high_quality else self.config.dpi
        
        fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        logger.info(f"Figure saved to {save_path}")


# Convenience functions for common use cases

def quick_binding_visualization(
    images: torch.Tensor,
    model_output: Dict[str, torch.Tensor],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Quick visualization of binding results for debugging.
    
    Args:
        images: Input images
        model_output: MVBA model output
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    visualizer = MVBAVisualizer()
    return visualizer.plot_slot_assignments(
        images, 
        model_output['masks'],
        model_output['spatial_attention'],
        save_path=save_path
    )


def quick_training_plot(
    training_logs: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Quick training curve visualization.
    
    Args:
        training_logs: Training metrics over time
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    visualizer = MVBAVisualizer()
    return visualizer.plot_training_curves(training_logs, save_path=save_path)


def analyze_model_performance(
    model_outputs: List[Dict[str, torch.Tensor]],
    target_images: List[torch.Tensor],
    save_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of model performance with multiple visualizations.
    
    Args:
        model_outputs: List of model outputs
        target_images: List of target images
        save_directory: Optional directory to save all visualizations
        
    Returns:
        Dictionary with analysis results
    """
    visualizer = MVBAVisualizer()
    
    # Analyze attention patterns
    spatial_attentions = [output['spatial_attention'] for output in model_outputs]
    combined_attention = torch.cat(spatial_attentions, dim=0)
    
    attention_stats = visualizer.analyze_attention_patterns(
        combined_attention,
        save_path=f"{save_directory}/attention_analysis.png" if save_directory else None
    )
    
    # Analyze failure modes
    failure_analysis = visualizer.analyze_failure_modes(
        model_outputs,
        target_images,
        save_path=f"{save_directory}/failure_analysis.png" if save_directory else None
    )
    
    # Create reconstruction quality visualization
    if model_outputs and target_images:
        # Handle different input formats for target_images
        first_target = target_images[0]
        first_recon = model_outputs[0]['reconstruction']
        
        # Ensure target_images has correct shape
        if first_target.dim() == 3:
            first_target = first_target.unsqueeze(0)
        if first_recon.dim() == 3:
            first_recon = first_recon.unsqueeze(0)
        
        visualizer.plot_reconstruction_quality(
            first_target,
            first_recon,
            save_path=f"{save_directory}/reconstruction_quality.png" if save_directory else None
        )
    
    return {
        'attention_stats': attention_stats,
        'failure_analysis': failure_analysis
    }