"""
Feature Binding Module for MVBA

This module answers the question: "WHAT features belong to each object?"

After spatial binding tells us WHERE objects are, feature binding determines
WHAT visual properties (features) belong to each object.

The hypothesis is power-law enhancement:
- Regular features: All features treated equally
- Power-law enhanced: Strong features amplified, weak features suppressed to increase the contrast
- Result: Each object's unique characteristics become more obvious

Biological inspiration:
This is inspired by how neurons in the brain increase their firing rates when
attending to specific features -> "Binding by Firing Rate Enhancement."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FeatureBinding(nn.Module):
    """
    This module takes the spatial attention maps (WHERE objects are) and uses
    them to extract and enhance the features (WHAT) of each object.
    
    3-step process:
    
    1. Feature Weighting: 
       - Use spatial attention to focus on each object's region: "spotlight" 
    
    2. Power-law Enhancement:
       - Apply f' = sign(f) * |f|^a transformation
       - Strong features get stronger, weak features get weaker
       - Creates clearer feature separation between objects
    
    3. Feature Aggregation:
       - Pool enhanced features for each object
       - Create a compact representation of what each object looks like
    
    Why power-law?
    - Linear scaling treats all features equally
    - Power-law creates sharpening
    - Simulates biological attention mechanisms in the brain 
        (enhancement + suppression using the same mechanism at the same time = attention = power-law)
    
    Example: Red car and blue ball
    - Car slot: Red features enhanced, blue suppressed
    - Ball slot: Blue features enhanced, red suppressed
    - Result: Clear feature separation between objects
    """
    
    def __init__(
        self,
        feature_dim: int = 72,
        slot_dim: int = 128,
        n_heads: int = 4,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize Feature Binding module.
        
        Args:
            feature_dim: Dimension of input features
            slot_dim: Dimension of slot representations
            n_heads: Number of attention heads for feature processing
            mlp_hidden_dim: Hidden dimension for MLP processing
            dropout: Dropout rate for regularization
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validate inputs
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be positive, got {slot_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {mlp_hidden_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.feature_dim = feature_dim
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        
        # === Feature Processing Network ===
        # Project raw extracted features into a higher dimension for binding
        self.feature_processor = nn.Sequential(
            # Expand features to higher dimension
            nn.Linear(feature_dim, mlp_hidden_dim),   # 72 -> 256
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),                      # Prevent overfitting
            # Project back to original dimension
            nn.Linear(mlp_hidden_dim, feature_dim)    # 256 -> 72
        )
        
        # === Slot-Conditioned Gating ===
        # Each slot has the ability selectively filter features
        self.slot_gate = nn.Sequential(
            # Input: concatenated slot state + features
            nn.Linear(slot_dim + feature_dim, mlp_hidden_dim),      # 128 + 72 -> 256
            nn.ReLU(inplace=True),
            # Output: gate values in [0, 1] for each feature
            nn.Linear(mlp_hidden_dim, feature_dim),                 # 256 -> 72
            nn.Sigmoid()  # 0 = block feature, 1 = keep feature
        )
        
        # === Feature-to-Slot Projection ===
        # Projects visual features to slot representation
        self.to_slot = nn.Sequential(
            nn.Linear(feature_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(slot_dim, slot_dim) 
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) 
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def apply_power_law_enhancement(
        self,
        features: torch.Tensor,
        alpha: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Apply power-law enhancement to features.
        
        This is the core power-law transformation:
        f' = sign(f) * |f|^a
        
        - a = 1: No change (identity transformation)
        - a > 1: Enhancement (amplifies differences)
        
        Preserve sign:
        - Feature representations can be negative 
        - We want to enhance magnitude while keeping direction
        
        Args:
            features: Features to enhance (B, n_slots, C, H, W)
                    B = batch, 
                    n_slots = objects, 
                    C = channels, 
                    H = height, 
                    W = width
            alpha: Enhancement factor (B, n_slots, 1)
                   Each slot has its own enhancement level
            eps: Small constant to prevent log(0) or division by zero
            
        Returns:
            Enhanced features with same shape as input
        """
        B, n_slots, C, H, W = features.shape
        
        # Expand alpha to broadcast with all feature dimensions
        # (B, n_slots, 1) -> (B, n_slots, 1, 1, 1)
        alpha_expanded = alpha.view(B, n_slots, 1, 1, 1)
        
        # Step 1: Separate sign and magnitude
        # sign: -1, 0, or +1 for each feature (B, n_slots, C, H, W)
        sign = torch.sign(features)
        # magnitude: absolute value (always positive) (B, n_slots, C, H, W)
        magnitude = torch.abs(features) + eps  # Add eps to avoid log(0)
        
        # Step 2: Apply power law using log-exp trick
        # log-exp is more numerically stable than direct power
        # |f|^a = exp(a × log(|f|))
        # (B, n_slots, 1, 1, 1) * log(B, n_slots, C, H, W) -> (B, n_slots, C, H, W)
        enhanced_magnitude = torch.exp(alpha_expanded * torch.log(magnitude))
        
        # Step 3: Normalize to prevent feature explosion
        # Normalization technique: Divide by mean 
        # Mean across C, H, W dimensions -> (B, n_slots, 1, 1, 1)
        mean_magnitude = enhanced_magnitude.mean(dim=(2, 3, 4), keepdim=True) + eps
        # (B, n_slots, C, H, W) / (B, n_slots, 1, 1, 1) -> (B, n_slots, C, H, W)
        enhanced_norm = enhanced_magnitude / mean_magnitude
        
        # Step 4: Restore direction
        # (B, n_slots, C, H, W) * (B, n_slots, C, H, W) -> (B, n_slots, C, H, W)
        enhanced = sign * enhanced_norm
        
        return enhanced
    
    def forward(
        self,
        features: torch.Tensor,
        spatial_attention: torch.Tensor,
        alphas: Dict[str, torch.Tensor],
        slots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature binding with power-law enhancement.
        
        Args:
            features: Visual features (B, C, H, W)
            spatial_attention: Spatial binding maps (B, n_slots, H, W)
            alphas: Dictionary with 'feature' key containing alpha values (B, n_slots, 1)
            slots: slot states for (B, n_slots, slot_dim)
             
        Returns:
            Tuple of:
            - Bound features per slot (B, n_slots, slot_dim)
            - Enhanced feature maps (B, n_slots, C, H, W)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # Defensive Porgramming: Validate inputs
        if features.dim() != 4:
            raise ValueError(f"Expected 4D features tensor, got {features.dim()}D")
        if spatial_attention.dim() != 4:
            raise ValueError(f"Expected 4D spatial_attention tensor, got {spatial_attention.dim()}D")
        if 'feature' not in alphas:
            raise ValueError("alphas must contain 'feature' key")
        
        B, C, H, W = features.shape
        B_s, n_slots, H_s, W_s = spatial_attention.shape
        
        if B != B_s:
            raise ValueError(f"Batch size mismatch: features {B} vs spatial_attention {B_s}")
        if C != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {C}")
        if (H, W) != (H_s, W_s):
            raise ValueError(f"Spatial size mismatch: features {(H, W)} vs attention {(H_s, W_s)}")
        
        feature_alpha = alphas['feature']
        if feature_alpha.shape != (B, n_slots, 1):
            raise ValueError(f"Expected feature alpha shape {(B, n_slots, 1)}, got {feature_alpha.shape}")
        
        if slots is not None:
            if slots.shape != (B, n_slots, self.slot_dim):
                raise ValueError(f"Expected slots shape {(B, n_slots, self.slot_dim)}, got {slots.shape}")
        
        # === Step 1: Process Features ===
        # Transform features to make them more suitable for binding
        # Flatten spatial dimensions for processing
        # Note: 
            # The MLP processes each spatial location independently, so flattening allows efficient parallel processing.
            # Spatial order is preserved: pixels are flattened row-by-row [1,2,3,4...] and restored to the exact same positions.
            # This is purely for computational efficiency - no spatial information is lost.
        features_flat = features.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        processed_features = self.feature_processor(features_flat)
        # Restore spatial dimensions
        processed_features = processed_features.transpose(1, 2).view(B, C, H, W)
        
        # === Step 2: Expand Features for Each Slot ===
        # Create a copy of features for each slot to process independently
        # (B, C, H, W) -> (B, n_slots, C, H, W)
        features_expanded = processed_features.unsqueeze(1).expand(B, n_slots, C, H, W)
        
        # === Step 3: Apply Spatial Attention ===
        # Weight features by WHERE each object is located
        spatial_attention_expanded = spatial_attention.unsqueeze(2)         # (B, n_slots, H, W) -> (B, n_slots, 1, H, W)
        # Main weighting + small residual (0.1) to preserve gradients
        # Residual prevents complete suppression of background features
        weighted_features = features_expanded * spatial_attention_expanded + features_expanded * 0.1
        
        # === Step 4: Apply Power-Law Enhancement ===
        # Enhance distinctive features
        enhanced_features = self.apply_power_law_enhancement(weighted_features, feature_alpha)
        
        # === Step 5: Aggregate Features Per Slot ===
        # Convert spatially distributed features into compact slot representations (pooling stage)
        bound_features_list = []
        
        for slot_idx in range(n_slots):
            # Extract this slot's enhanced features and attention
            slot_features = enhanced_features[:, slot_idx]  # (B, C, H, W)
            slot_attention = spatial_attention[:, slot_idx]  # (B, H, W)
            
            # Prepare for weighted pooling
            slot_features_flat = slot_features.view(B, C, H * W)  # (B, C, H*W)
            attention_weights = slot_attention.view(B, H * W).unsqueeze(1)  # (B, 1, H*W)
            
            # Normalize attention weights with temperature
            # Temperature = 0.1 makes the attention sharper (more focused)
            # This ensures features come mainly from high-attention regions
            attention_weights = F.softmax(attention_weights / 0.1, dim=-1)
            
            # Weighted pooling: sum features weighted by attention
            # bmm = batch matrix multiply: (B, 1, H*W) x (B, H*W, C) = (B, 1, C)
            # Result: weighted average of features across spatial locations
            pooled_features = torch.bmm(attention_weights, slot_features_flat.transpose(1, 2))  # (B, 1, C)
            pooled_features = pooled_features.squeeze(1)  # (B, C)
            
            # === Slot-Conditioned Gating ===
            # Let slots selectively filter features based on their state
            if slots is not None:
                slot_state = slots[:, slot_idx]  # (B, slot_dim)
                # Concatenate slot state with pooled features
                gate_input = torch.cat([slot_state, pooled_features], dim=-1)
                # Compute gate values (0 to 1 for each feature)
                gate = self.slot_gate(gate_input)
                # Apply gating
                pooled_features = pooled_features * gate
            
            # === Step 6: Project to Slot Dimension ===
            # Bind features to slots
            slot_repr = self.to_slot(pooled_features)  # (B, slot_dim)
            bound_features_list.append(slot_repr)
        
        # Stack all slot representations into single tensor
        bound_features = torch.stack(bound_features_list, dim=1)  # (B, n_slots, slot_dim)
        
        return bound_features, enhanced_features
    
    def compute_feature_diversity(
        self,
        bound_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diversity of bound features across slots.
        
        This metric tells us: "How different are the objects from each other?"
        - High diversity: Each slot represents something different (good!)
        - Low diversity: Slots are redundant, representing similar things (bad!)
        
        The computation:
        1. Normalize features (to compare directions, not magnitudes)
        2. Compute cosine similarity between all slot pairs
        3. Average similarity (excluding self-similarity)
        4. Convert to diversity score (1 - similarity)
        
        Example scores:
        - Diversity = 1.0: Objects are very different (ideal)
        - Diversity = 0.5: Objects are somewhat similar
        - Diversity = 0.0: Objects are identical (problematic)
        
        Args:
            bound_features: Feature representations for each slot (B, n_slots, slot_dim)
                           These are the "what" descriptions of each object
            
        Returns:
            Diversity score for each batch item (B,)
            Higher scores = more diverse objects = better binding
        """
        B, n_slots, _ = bound_features.shape
        
        # Step 1: Normalize features to unit length
        # This makes cosine similarity = dot product
        # We care about direction (what kind of object) not magnitude
        features_norm = F.normalize(bound_features, p=2, dim=-1)
        
        # Step 2: Compute pairwise cosine similarity
        # bmm performs: features_norm @ features_norm.T for each batch
        # Result[i,j] = similarity between slot i and slot j
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))  # (B, n_slots, n_slots)
        
        # Step 3: Mask diagonal (self-similarity)
        # A slot is always perfectly similar to itself (similarity = 1)
        # We only care about similarity between different slots
        mask = torch.eye(n_slots, device=similarity.device).unsqueeze(0).expand(B, -1, -1)
        similarity = similarity * (1 - mask)  # Zero out diagonal
        
        # Step 4: Compute average similarity
        # Sum all off-diagonal elements and divide by number of pairs
        n_pairs = n_slots * (n_slots - 1)  # Number of unique slot pairs
        avg_similarity = similarity.sum(dim=[1, 2]) / n_pairs  # (B,)
        
        # Step 5: Convert to diversity score
        # Similarity close to 0 = very different = high diversity
        # Similarity close to 1 = very similar = low diversity
        # abs() handles potential negative similarities
        diversity = 1 - avg_similarity.abs()
        
        return diversity
