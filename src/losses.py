"""
Loss Functions for MVBA Training

Implements Reconstruction loss needed for training the binding architecture:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss between input images and model reconstructions.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean"
    ):
        """
        Initialize reconstruction loss.
        
        Args:
            loss_type: Type of loss ("mse")
            reduction: Reduction method ("mean", "sum", "none")
            
        Raises:
            ValueError: If loss_type or reduction is invalid
        """
        super().__init__()
        
        valid_reductions = ["mean", "sum", "none"]
        if reduction not in valid_reductions:
            raise ValueError(f"reduction must be one of {valid_reductions}, got {reduction}")
        
        self.loss_type = loss_type
        self.reduction = reduction
        
        # Set up loss function
        self.loss_fn = nn.MSELoss(reduction=reduction)

    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            reconstruction: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)
            mask: Optional mask for weighted loss (B, 1, H, W)
            
        Returns:
            Reconstruction loss value
            
        Raises:
            ValueError: If shapes don't match
        """
        if reconstruction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {reconstruction.shape} vs {target.shape}")
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1:
                mask = mask.expand_as(reconstruction)
            
            # Apply mask
            reconstruction = reconstruction * mask
            target = target * mask
        
        return self.loss_fn(reconstruction, target)

class MVBALoss(nn.Module):
    """
    Combined loss function for MVBA training.
    
    """
    
    def __init__(
        self,
        # Reconstruction loss parameters
        recon_type: str = "mse",
        recon_weight: float = 1.0,
    ):
        """
        Initialize combined MVBA loss.
        
        Args:
            recon_type: Type of reconstruction loss
            recon_weight: Weight for reconstruction loss
        """
        super().__init__()
        
        # Initialize components
        self.reconstruction = ReconstructionLoss(loss_type=recon_type)
        
        # Weights
        self.recon_weight = recon_weight
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined MVBA loss.
        
        Args:
            reconstruction: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            Dictionary with:
            - 'total': Total combined loss
            - 'reconstruction': Reconstruction loss
        """
        # Compute individual losses
        recon_loss = self.reconstruction(reconstruction, target) * self.recon_weight
        
        # Total loss
        total_loss = recon_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
        }