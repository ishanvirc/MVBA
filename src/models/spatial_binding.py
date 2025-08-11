"""
Spatial Binding Module for MVBA

"WHERE is each object located?"

This module provides a map for each slot showing which pixels belong to its slot.
Reference: This module is an extension of the Standard Slot Attention's Spatial Binding Module (Locatello et al., 2020).

MVBA contributions:  
1. Alpha sharpening integration.
   - Power-law sharpening before softmax
    - Enhances strong matches, suppresses weak ones (sharpening)
2. Multi-scale refinement (gestalt grouping) 
   - Purpose: Create more coherent, blob-like regions 
   - Uses 3 different kernel sizes (3x3, 5x5, 7x7)
   - Each scale captures different spatial patterns
   - Learnable convolutional layers refine attention maps

The process:
1. Slots generate spatial queries: "I'm looking for regions like this"
2. Image features provide keys: "Here's what's at each location"  
3. Matching queries to keys creates attention maps
4. Alpha sharpens the attention maps
5. Multi-Scale Refinement enforces spatial coherence
6. Softmax creates competition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class SpatialBinding(nn.Module):
    """
    SpatialBinding: Determines WHERE each object is located in the image.
    
    This module creates spatial attention maps that show which pixels belong
    to which slot/object. It's the spatial component of the binding problem.
    
    Key features:
    
    1. Spatial query generation:
       - Slots produce queries: "I'm looking for these spatial patterns"
       - Similar to asking "Where are the regions that match my object?"
    
    2. Query-key matching:
       - Features at each location provide keys
       - High match score = "This location belongs to this slot"
    
    3. Alpha enhancement (The innovation):
       - Applied to logits BEFORE softmax
       - Sharpens matching: weak matches -> weaker, strong matches -> stronger
    
    4. Multi-scale refinement:
       - Uses different kernel sizes (3x3, 5x5, 7x7)
       - Captures patterns at different scales
       - Helps create coherent, blob-like regions (Gestalt grouping)
    
    5. Competitive softmax:
       - Normalizes across slots, not spatial locations
    """
    
    def __init__(
        self,
        slot_dim: int = 128,
        feature_dim: int = 72,
        query_dim: int = 32,
        n_refinement_layers: int = 3
    ):
        """
        Initialize Spatial Binding module.
        
        Args:
            slot_dim: Dimension of slot representations
            feature_dim: Dimension of input features (with positional encoding)
            query_dim: Dimension for query-key matching
            n_refinement_layers: Number of convolutional refinement layers
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Defensive Programming: Validate inputs
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be positive, got {slot_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive, got {query_dim}")
        if n_refinement_layers < 0:
            raise ValueError(f"n_refinement_layers must be non-negative, got {n_refinement_layers}")
        
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.query_dim = query_dim
        self.n_refinement_layers = n_refinement_layers
        self.eps = 1e-8  # For numerical stability in power-law enhancement
        
        # === Spatial Query Network ===
        # Transforms slot representations into spatial queries
        # "What spatial patterns am I looking for?"
        self.to_spatial_q = nn.Sequential(
            # Compress slot information
            nn.Linear(slot_dim, slot_dim // 2),
            nn.LayerNorm(slot_dim // 2),
            nn.ReLU(inplace=True),
            # Project to query dimension for matching
            nn.Linear(slot_dim // 2, query_dim)
        )
        
        # === Spatial Key Network ===
        # Transforms image features into keys at each location
        # "What spatial information is available here?"
        # 1x1 conv = pointwise linear transformation at each spatial location
        self.to_spatial_k = nn.Conv2d(feature_dim, query_dim, kernel_size=1)
        
        # === Multi-Scale Refinement Network ===
        # Purpose: Clean up noisy attention maps to create coherent object regions
        #   - Raw attention maps can be noisy with scattered pixels. 
        #   - Objects in the real world are usually coherent blobs, not random scattered points. 
        #   - This network learns to group nearby similar pixels together (Gestalt principle).
        
        # Kernel sizes:
        # - 3x3 kernel: Fine brush for details (e.g., object edges)
        # - 5x5 kernel: Medium brush for local smoothing
        # - 7x7 kernel: Large brush for global coherence
        
        self.refinement_layers = nn.ModuleList()
        
        # Select first n kernel sizes from [3, 5, 7] based on n_refinement_layers (1 to 3)
        kernel_sizes = [3, 5, 7][:n_refinement_layers]
        
        for kernel_size in kernel_sizes:
            # Build a mini-network for each scale
            # This is a learnable image filter that enfoces coherence attention maps based on spatial context
            
            layer = nn.Sequential(
                # First conv: 1 -> 16 channels
                # Input: Single-channel attention map for one slot
                # Output: 16 feature maps detecting different spatial patterns
                nn.Conv2d(1, 16, kernel_size, padding=(kernel_size-1)//2),
                
                # GroupNorm: Normalize within groups for stability
                # Splits 16 channels into 4 groups of 4 channels each
                nn.GroupNorm(4, 16),
                
                # Allows network to learn non-linear refinement patterns
                nn.ReLU(inplace=True),
                
                # Second conv: 16 -> 1 channel  
                # Combines all 16 pattern detectors into final refinement
                # Output will be added to original attention as residual
                nn.Conv2d(16, 1, kernel_size, padding=(kernel_size-1)//2)
            )
            
            # padding=(kernel_size-1)//2 ensures "same" convolution:
            # - kernel_size=3: padding=(3-1)//2=1
            # - kernel_size=5: padding=(5-1)//2=2  
            # - kernel_size=7: padding=(7-1)//2=3
            # This preserves spatial dimensions: H_out = H_in, W_out = W_in
            
            self.refinement_layers.append(layer)
        
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for stable training.
        
        Xavier initialization considers the number of input and output connections to keep variance constant.
        It helps prevent the vanishing/exploding gradient problem during training by ensuring that the variance 
        of activations and gradients remains relatively consistent across layers

        The networks are relatively shallow and Xavier works well in practice, no need for a complex initialization scheme.
        """
        
        # === Initialize Query Network ===
        for module in self.to_spatial_q.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm parameters:
                # - weight: Initialize to 1 (no scaling initially)
                # - bias: Initialize to 0   (no shift initially)
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # === Initialize Key Network ===
        nn.init.xavier_uniform_(self.to_spatial_k.weight)
        # Using Xavier for keys as well, because we need keys and queries to have similar scales
        if self.to_spatial_k.bias is not None:
            nn.init.constant_(self.to_spatial_k.bias, 0)
        
        # === Initialize Refinement Layers ===
        if self.n_refinement_layers > 0:
            for layer in self.refinement_layers:
                for module in layer.modules():
                    if isinstance(module, nn.Conv2d):
                        # Xavier for conv layers too
                        # Especially important here because we're learning filters
                        # Bad init could lead to all-zero or exploding refinements
                        nn.init.xavier_uniform_(module.weight)
                        
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                            
                    elif isinstance(module, nn.GroupNorm):
                        # GroupNorm initialization (same as LayerNorm)
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
        alphas: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute spatial binding fields with alpha enhancement.
        
        Args:
            slots:    (B, n_slots, slot_dim)
            features: (B, C, H, W)
            alphas: Dictionary with 'spatial' key containing alpha values (B, n_slots, 1)
            
        Returns:
            Spatial attention maps of shape (B, n_slots, H, W)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # Defensive Programming: Validate inputs
        if slots.dim() != 3:
            raise ValueError(f"Expected 3D slots tensor, got {slots.dim()}D")
        if features.dim() != 4:
            raise ValueError(f"Expected 4D features tensor, got {features.dim()}D")
        if 'spatial' not in alphas:
            raise ValueError("alphas must contain 'spatial' key")
        
        B, n_slots, slot_dim = slots.shape
        B_f, C, H, W = features.shape
        
        if B != B_f:
            raise ValueError(f"Batch size mismatch: slots {B} vs features {B_f}")
        if slot_dim != self.slot_dim:
            raise ValueError(f"Expected slot_dim={self.slot_dim}, got {slot_dim}")
        if C != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {C}")
        
        # === Step 1: Generate Queries and Keys ===
        # Slots produce queries: "What spatial patterns am I looking for?"
        q = self.to_spatial_q(slots)  # (B, n_slots, query_dim)
        
        # Features produce keys: "What's at each spatial location?"
        k = self.to_spatial_k(features)  # (B, query_dim, H, W)
        
        # === Step 2: Compute Attention Scores (Logits) ===
        
        # Reshape keys for efficient batch matrix multiplication
        # (B, query_dim, H, W) -> (B, query_dim, H*W)
        k_flat = k.view(B, self.query_dim, H * W)
        
        # Compute dot product between queries and keys
        # High score = good match between what slot wants and what's there
        # bmm = batch matrix multiply: q @ k for each image in batch
        # The matrix multiplication performs: (B, n_slots, query_dim) x (B, query_dim, H*W) -> (B, n_slots, H*W)
        logits = torch.bmm(q, k_flat)  #(B, n_slots, H*W)
        
        # Reshape back to spatial format
        logits = logits.view(B, n_slots, H, W)
        
        # === Step 3: Apply Alpha Scaling ===
        spatial_alpha = alphas['spatial']  # (B, n_slots, 1)
        if spatial_alpha.shape != (B, n_slots, 1):
            raise ValueError(f"Expected alpha shape {(B, n_slots, 1)}, got {spatial_alpha.shape}")
        
        # Expand alpha to broadcast with spatial dimensions
        # (B, n_slots, 1) -> (B, n_slots, 1, 1)
        # This allows element-wise multiplication with (B, n_slots, H, W)
        alpha_expanded = spatial_alpha.view(B, n_slots, 1, 1)
        
        # Apply power-law enhancement
        # This creates non-linear enhancement where stronger logits get exponentially stronger (sharpening)
        # Slots with high alpha become exponentially more "decisive" in their claims
        
        # Step 3.1: Apply power-law to magnitude while preserving sign
        enhanced_logits = torch.sign(logits) * torch.pow(torch.abs(logits) + self.eps, alpha_expanded)
        
        # Step 3.2: Normalize by mean magnitude to maintain stable scale
        mean_magnitude = enhanced_logits.abs().mean(dim=(2, 3), keepdim=True) + self.eps
        scaled_logits = enhanced_logits / mean_magnitude
        
        # === Step 4: Multi-Scale Refinement ===
        # Purpose: Transform noisy attention into clean object segments.
        # Nearby pixels that are similar should belong to the same object
        
        # Clone to seperate refined logits from original (code clarity)
        refined_logits = scaled_logits.clone()
        
        # Process each slot's attention map separately, each slot might need different refinement
        # (e.g., small objects need fine details, large objects need smoothing)
        for i in range(n_slots):
            # Extract one slot's logits (keep dims for conv layers)
            # Using slicing notation i:i+1 instead of indexing i preserves the slot dimension as 1
            # - This gives us (B, 1, H, W) instead of (B, H, W)
            # Conv2d expects 4D input: (batch, channels, height, width)
            slot_logits = scaled_logits[:, i:i+1]  # (B, 1, H, W)
            
            # Apply each scale of refinement sequentially, each builds on the previous
            for layer in self.refinement_layers:
                # Compute refinement delta
                # The network learns what changes to make:
                # Examples:
                # - Fill holes in objects
                # - Smooth jagged edges  
                # - Suppress isolated pixels
                # - Enhance coherent regions
                refinement = layer(slot_logits)
                
                # Add as small residual (weight = 0.1)
                # Small: We want to refine, not replace
                # Residual: Preserves original evidence while improving it
                slot_logits = slot_logits + 0.1 * refinement
                
                # The cumulative effect over multiple scales:
                # - 3x3: Cleans up pixel-level noise
                # - 5x5: Smooths local regions
                # - 7x7: Ensures global coherence
            
            # Put refined logits back (remove channel dimension)
            # squeeze(1) removes the channel dim: (B, 1, H, W) -> (B, H, W)
            # We need this because refined_logits[:, i] expects (B, H, W) shape
            refined_logits[:, i] = slot_logits.squeeze(1)
        
        # Use refined logits for final attention computation
        scaled_logits = refined_logits
        
        # === Step 5: Competitive Softmax ===
        # slots compete for pixels!
        # 
        # We normalize across SLOTS (dim=1), not spatial locations
        # This means at each pixel, the slots compete and sum to 1
        # Result: Each pixel assignment is normalized and is assigned PRIMARILY to ONE slot. 
        # 
        # Example at one pixel:
        # Before softmax: Slot1=3.0, Slot2=1.0, Slot3=0.5
        # After softmax:  Slot1=0.82, Slot2=0.11, Slot3=0.07
        # Slot 1 wins this pixel!
        
        spatial_attention = F.softmax(scaled_logits, dim=1)  # (B, n_slots, H, W)
        
        return spatial_attention
    

    def compute_binding_entropy(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of spatial binding for monitoring.
        
        The math:
        Entropy = -Σ p * log(p)
        where p is the probability of each slot at each location
        
        Args:
            spatial_attention: Spatial attention maps (B, n_slots, H, W)
                              Values are probabilities that sum to 1 across slots
            
        Returns:
            Average entropy values for each image in batch (B,)
            Lower values indicate better, more decisive binding
        """
        # Add small epsilon to prevent log(0) which would give -inf
        eps = 1e-8
        log_attention = torch.log(spatial_attention + eps)
        
        # Compute entropy at each spatial location
        # Entropy formula: H = -Σ p * log(p)
        # Sum across slots (dim=1) to get entropy at each pixel
        entropy = -(spatial_attention * log_attention).sum(dim=1)  # (B, H, W)
        
        # Average entropy across all spatial locations
        # This gives us a single entropy value per image
        mean_entropy = entropy.mean(dim=[1, 2])  # (B,)
        
        # Interpretation:
        # - In practice, good binding gives entropy < 0.5
        
        return mean_entropy