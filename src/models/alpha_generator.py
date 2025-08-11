"""
Alpha Generation Module for MVBA

This module generates the alpha values (spatial + feature) 

What is Alpha?
- Alpha is a sharpening parameter that controls competition between slots
- Higher alpha = more winner-take-all behavior (sharper boundaries)
- Lower alpha = softer competition (more sharing between objects)

Design choices:
1. Alpha depends only on slot state (top-down signal):
- The slot decides its confidence based on what it represents
- NOT based on current image features (that would be bottom-up)
- This prevents circular dependencies in the computation and improves biological plausibility

2. Separate networks for spatial and feature alpha:
- Spatial alpha: Controls pixel assignment sharpness
- Feature alpha: Controls feature enhancement strength
- Different tasks may need different sharpness levels

3. Constrained to reasonable range [min_alpha, max_alpha]:
- Too low (= 1): No enhancement
- Too high (> 5): Numerical instability, gradient problems
- Sweet spot: Between 1.0 and 3.0

4. Smooth sigmoid-based scaling:
- Network outputs sigmoid (0 to 1)
- Scaled to [min_alpha, max_alpha]
- Designed this way to solve gradient problems for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class AlphaGenerator(nn.Module):
    """
    This module learns to generate alpha values that control how strongly slots compete.
    - Alpha = 1: Fair competition (like regular softmax)
    - Alpha = 2: Moderately greedy (clearer winners)
    - Alpha = 3+: Very greedy (winner takes almost all)
    """
    
    def __init__(
        self,
        slot_dim: int = 128,
        min_alpha: float = 1.0,
        max_alpha: float = 3.0,
        hidden_dim: int = 64
    ):
        """
        Initialize Alpha Generator.
        
        Args:
            slot_dim: Dimension of slot representations
            min_alpha: Minimum alpha value 
            max_alpha: Maximum alpha value 
            hidden_dim: Hidden dimension for alpha networks
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Defensive Programming: Validate input parameters
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be positive, got {slot_dim}")
        if min_alpha < 0.1:
            raise ValueError(f"min_alpha must be >= 0.1, got {min_alpha}")
        if max_alpha <= min_alpha:
            raise ValueError(f"max_alpha must be > min_alpha, got max={max_alpha}, min={min_alpha}")
        if max_alpha > 10.0:
            raise ValueError(f"max_alpha should be <= 10.0 for numerical stability, got {max_alpha}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        self.slot_dim = slot_dim
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.alpha_range = max_alpha - min_alpha
        
        # === Spatial Alpha Network ===
        # This network learns to generate alpha values for spatial competition in spatial_binding.py
        # Input: slot state -> Output: alpha parameter
        self.alpha_spatial = nn.Sequential(
            # Layer 1: Project slot state to hidden dimension
            nn.Linear(slot_dim, hidden_dim),
            # LayerNorm: Normalizes activations for stable training
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Layer 2: Compress representation
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Output layer: Single alpha value
            nn.Linear(hidden_dim // 2, 1),
            # Sigmoid: Ensures output is in [0, 1] range
            nn.Sigmoid()
            # Will be scaled to [min_alpha, max_alpha] later
        )
        
        # === Feature Alpha Network ===
        # This network learns to generate alpha values for feature competition in feature_binding.py
        self.alpha_feature = nn.Sequential(
            # Same architecture as spatial network
            # But with independent weights - can learn different strategies
            nn.Linear(slot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize network weights.
        
        Goal: Start with moderate alpha values, this helps the model learn gradually without getting stuck.
        Design choice: Xavier uniform initialization, due to Sigmoid activation function used in alpha networks. 
        """
        for module in [self.alpha_spatial, self.alpha_feature]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    
                    if layer.bias is not None:
                        if layer.out_features == 1:  # Final layer (output layer)
                            # Special initialization for sigmoid output
                            # We want initial sigmoid output = 0.3
                            # This gives initial alpha = min_alpha + 0.3 * (max_alpha - min_alpha)
                            # For default range [1, 3], this gives alpha = 1.6
                            # 
                            # Math: sigmoid(x) = 1/(1 + e^(-x))
                            # To get sigmoid(x) = 0.3:
                            # 0.3 = 1/(1 + e^(-x))
                            # 1 + e^(-x) = 3.33
                            # e^(-x) = 2.33
                            # -x = ln(2.33) = 0.85
                            # x = -0.85
                            nn.init.constant_(layer.bias, -0.85)
                        else:
                            # Hidden layers: zero bias is standard
                            nn.init.constant_(layer.bias, 0)
                            
                elif isinstance(layer, nn.LayerNorm):
                    # Standard LayerNorm initialization
                    nn.init.constant_(layer.weight, 1)  # No scaling initially
                    nn.init.constant_(layer.bias, 0)    # No shift initially
    
    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate alpha values for each slot.
        
        The process:
        1. Each slot's state is processed by the alpha networks
        2. Networks output values in [0, 1] (sigmoid)
        3. We scale these to [min_alpha, max_alpha]
        4. Each slot gets its own alpha values for spatial and feature binding
        
        Args:
            slots: Slot representations of shape (B, n_slots, slot_dim)
                   - B: Batch size
                   - n_slots: Number of object slots
                   - slot_dim: Dimension of each slot's representation
            
        Returns:
            Dictionary containing:
            - 'spatial': Alpha values for spatial binding (B, n_slots, 1)
            - 'feature': Alpha values for feature binding (B, n_slots, 1)
            
        Raises:
            ValueError: If input shape is invalid
        """
        # Validate input shape
        if slots.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, n_slots, slot_dim), got {slots.dim()}D")
        
        B, n_slots, slot_dim = slots.shape
        
        if slot_dim != self.slot_dim:
            raise ValueError(f"Expected slot_dim={self.slot_dim}, got {slot_dim}")
        
        # === Step 1: Reshape for efficient batch processing ===
        # Instead of looping over slots, process all at once: all slots across all images get their alpha values computed simulataneously (GPU parallelization)
        # (B, n_slots, slot_dim) -> (B*n_slots, slot_dim)
        slots_flat = slots.reshape(B * n_slots, slot_dim)
        
        # === Step 2: Generate normalized alpha values ===
        # Networks output sigmoid values in [0, 1]
        spatial_norm = self.alpha_spatial(slots_flat)  # (B*n_slots, 1)
        feature_norm = self.alpha_feature(slots_flat)  # (B*n_slots, 1)
        
        # === Step 3: Reshape back to separate slots ===
        # (B*n_slots, 1) -> (B, n_slots, 1)
        spatial_norm = spatial_norm.reshape(B, n_slots, 1)
        feature_norm = feature_norm.reshape(B, n_slots, 1)
        
        # === Step 4: Scale to alpha range ===
        # Transform [0, 1] -> [min_alpha, max_alpha]
        # Formula: min + normalized * (max - min)
        # [0, 1] -> [1.0, 3.0]
        alphas = {
            'spatial': self.min_alpha + spatial_norm * self.alpha_range,
            'feature': self.min_alpha + feature_norm * self.alpha_range
        }
        
        return alphas
    
    def get_alpha_stats(self, slots: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics about generated alpha values.
        
        Args:
            slots: Slot representations
            
        Returns:
            Dictionary with statistics for each alpha type:
            - {name}_mean: Average alpha value
            - {name}_std: Standard deviation (variation between slots)
            - {name}_min: Minimum alpha value
            - {name}_max: Maximum alpha value
            
        Example output:
            {
                'spatial_alpha_mean': 1.8,
                'spatial_alpha_std': 0.3,
                'spatial_alpha_min': 1.2,
                'spatial_alpha_max': 2.4,
                'feature_alpha_mean': 2.1,
                ...
            }
        """
        # Compute without tracking gradients (more efficient)
        with torch.no_grad():
            alphas = self.forward(slots)
            
            stats = {}
            for name, alpha_vals in alphas.items():
                # Flatten the statistics into a single dictionary
                # This makes it easier to log and monitor
                stats[f'{name}_alpha_mean'] = alpha_vals.mean().item()
                stats[f'{name}_alpha_std'] = alpha_vals.std().item()
                stats[f'{name}_alpha_min'] = alpha_vals.min().item()
                stats[f'{name}_alpha_max'] = alpha_vals.max().item()
            
            return stats