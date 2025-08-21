"""
Validation Metrics for MVBA Architecture

This module provides metrics for evaluating the quality of object binding in the
Minimal Viable Binding Architecture (MVBA).
- Reconstruction quality: PSNR, SSIM, MSE, L1
- Slot utilization: Slot diversity
- Binding consistency: Spatial consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Optional


class MVBAMetrics:
    """
    Simplified metrics suite for MVBA training and evaluation.
    Only includes metrics actually used in the training pipeline.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        eps: float = 1e-8,
        compute_expensive: bool = True
    ):
        """
        Initialize metrics calculator.
        
        Args:
            device: Device for computations (defaults to input tensor device)
            eps: Small epsilon for numerical stability
            compute_expensive: Whether to compute SSIM (set False for faster training)
        """
        self.device = device
        self.eps = eps
        self.compute_expensive = compute_expensive
    
    def compute_all_metrics(
        self,
        model_output: Dict[str, torch.Tensor],
        target_images: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        return_individual: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all essential metrics for MVBA model output.
        
        Args:
            model_output: Dictionary from MVBA forward pass containing:
                - 'reconstruction': Reconstructed images (B, C, H, W)
                - 'slots': Slot representations (B, n_slots, slot_dim)
                - 'spatial_attention': Spatial binding maps (B, n_slots, H, W)
                - 'bound_features': Bound feature representations (B, n_slots, slot_dim)
            target_images: Ground truth images (B, C, H, W)
            target_masks: Optional ground truth masks (not used in simplified version)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary containing computed metrics organized by category
        """
        metrics = {}
        
        # 1. Reconstruction Quality Metrics
        reconstruction = model_output['reconstruction']
        metrics['reconstruction_quality'] = self.compute_reconstruction_quality(
            reconstruction, target_images, return_individual
        )
        
        # 2. Slot Utilization Metrics (only diversity is used)
        slots = model_output['slots']
        metrics['slot_utilization'] = {
            'slot_diversity': self.compute_slot_diversity(slots, return_individual)
        }
        
        # 3. Binding Consistency Metrics (only spatial consistency is used)
        spatial_attention = model_output['spatial_attention']
        metrics['binding_consistency'] = {
            'spatial_consistency': self.compute_spatial_consistency(
                spatial_attention, return_individual
            )
        }
        
        return metrics
    
    def compute_reconstruction_quality(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Compute reconstruction quality metrics.
        
        Args:
            reconstruction: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary with reconstruction metrics
        """
        # MSE
        mse_scores = F.mse_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))
        
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr_scores = 20 * torch.log10(1.0 / torch.sqrt(mse_scores + self.eps))
        
        # L1 loss
        l1_scores = F.l1_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))
        
        # SSIM (if enabled)
        if self.compute_expensive:
            ssim_scores = self._compute_ssim(reconstruction, target)
        else:
            ssim_scores = torch.zeros_like(mse_scores)
        
        if return_individual:
            return {
                'mse': mse_scores,
                'psnr': psnr_scores,
                'l1': l1_scores,
                'ssim': ssim_scores
            }
        else:
            return {
                'mse': mse_scores.mean().item(),
                'psnr': psnr_scores.mean().item(),
                'l1': l1_scores.mean().item(),
                'ssim': ssim_scores.mean().item()
            }
    
    def compute_slot_diversity(
        self,
        slots: torch.Tensor,
        return_individual: bool = False
    ) -> Union[float, torch.Tensor]:
        """
        Compute diversity of slot representations.
        High diversity means slots are learning different features.
        
        Args:
            slots: Slot representations (B, n_slots, slot_dim)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Slot diversity score(s)
        """
        B, n_slots, slot_dim = slots.shape
        diversity_scores = []
        
        for b in range(B):
            slot_features = slots[b]  # (n_slots, slot_dim)
            
            # Compute pairwise cosine similarities
            slot_features_norm = F.normalize(slot_features, dim=-1)
            similarity_matrix = torch.mm(slot_features_norm, slot_features_norm.t())
            
            # Remove diagonal (self-similarities)
            mask = ~torch.eye(n_slots, dtype=torch.bool, device=slots.device)
            off_diagonal_sims = similarity_matrix[mask]
            
            # Diversity = 1 - average similarity (higher = more diverse)
            diversity = 1.0 - off_diagonal_sims.mean()
            diversity_scores.append(diversity)
        
        diversity_scores = torch.stack(diversity_scores)
        
        if return_individual:
            return diversity_scores
        else:
            return diversity_scores.mean().item()
    
    def compute_spatial_consistency(
        self,
        spatial_attention: torch.Tensor,
        return_individual: bool = False
    ) -> Union[float, torch.Tensor]:
        """
        Compute spatial consistency of object bindings.
        Measures how focused/consistent each slot's attention is.
        
        Args:
            spatial_attention: Spatial binding maps (B, n_slots, H, W)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Spatial consistency score(s)
        """
        B, n_slots, H, W = spatial_attention.shape
        consistency_scores = []
        
        for b in range(B):
            batch_consistency = []
            for s in range(n_slots):
                attention_map = spatial_attention[b, s]  # (H, W)
                
                # Normalize attention to probability distribution
                attention_flat = attention_map.flatten()
                attention_norm = attention_flat / (attention_flat.sum() + self.eps)
                
                # Compute entropy (lower entropy = more focused = higher consistency)
                entropy = -(attention_norm * torch.log(attention_norm + self.eps)).sum()
                max_entropy = np.log(H * W)
                
                # Convert to consistency score (1 = perfectly focused, 0 = uniform)
                consistency = 1.0 - (entropy / max_entropy)
                batch_consistency.append(consistency)
            
            # Average consistency across all slots
            consistency_scores.append(torch.stack(batch_consistency).mean())
        
        consistency_scores = torch.stack(consistency_scores)
        
        if return_individual:
            return consistency_scores
        else:
            return consistency_scores.mean().item()
    
    def _compute_ssim(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Structural Similarity Index (simplified implementation).
        
        Args:
            reconstruction: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            SSIM scores per batch sample
        """
        B, C, H, W = reconstruction.shape
        ssim_scores = []
        
        for b in range(B):
            recon = reconstruction[b]
            tgt = target[b]
            
            # Convert to grayscale if RGB
            if C == 3:
                recon = 0.299 * recon[0] + 0.587 * recon[1] + 0.114 * recon[2]
                tgt = 0.299 * tgt[0] + 0.587 * tgt[1] + 0.114 * tgt[2]
            else:
                recon = recon[0]
                tgt = tgt[0]
            
            # Compute mean and variance
            mu1 = recon.mean()
            mu2 = tgt.mean()
            
            sigma1_sq = ((recon - mu1) ** 2).mean()
            sigma2_sq = ((tgt - mu2) ** 2).mean()
            sigma12 = ((recon - mu1) * (tgt - mu2)).mean()
            
            # SSIM formula constants
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            # Compute SSIM
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            ssim_scores.append(ssim)
        
        return torch.stack(ssim_scores)


def quick_metrics(
    model_output: Dict[str, torch.Tensor],
    target_images: torch.Tensor
) -> Dict[str, float]:
    """
    Quick metrics computation for training monitoring.
    Returns only the essential scalar metrics.
    
    Args:
        model_output: Dictionary from MVBA forward pass
        target_images: Ground truth images
        
    Returns:
        Dictionary with key scalar metrics
    """
    metrics_calc = MVBAMetrics(compute_expensive=False)  # Skip SSIM for speed
    all_metrics = metrics_calc.compute_all_metrics(
        model_output, target_images, return_individual=False
    )
    
    # Flatten to single dictionary for easy logging
    return {
        'psnr': all_metrics['reconstruction_quality']['psnr'],
        'mse': all_metrics['reconstruction_quality']['mse'],
        'l1': all_metrics['reconstruction_quality']['l1'],
        'slot_diversity': all_metrics['slot_utilization']['slot_diversity'],
        'spatial_consistency': all_metrics['binding_consistency']['spatial_consistency']
    }
