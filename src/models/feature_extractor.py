"""
Feature Extraction Layer for MVBA

Key concepts:
- CNN (Convolutional Neural Network): Extracts visual patterns
- Positional encoding: Adds location information to features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FeatureExtractor(nn.Module):
    """
    Feature Extractor: Converts images into bindable features.

    Key design choices explained:
    
    1. No Spatial Pooling:
       - No aggressive downsampling like classification networks
       - Keeping full resolution helps the model know WHERE features are located
       - Assumption: spatial information is crucial for binding features to the correct objects
    
    2. Learnable positional encoding:
       - Adds a "spatial signature" to features
       - Helps the model understand that features at (10,20) are different from the same features at (30,40)
    
    3. GroupNorm instead of BatchNorm:
       - flexible with any batch size
       - stable during training
       - doesn't depend on other images in the batch (but does so in the group)
    """
    
    def __init__(self, 
                in_channels: int = 3, 
                base_channels: int = 32, 
                out_channels: int = 64):
        """
        Initialize the feature extractor.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels for CNN
            out_channels: Number of output feature channels
        """
        super().__init__()
        
        # Validate inputs
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        
        # Assign parameters
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        
        # === CNN Backbone ===
        self.backbone = nn.Sequential(
            # Layer 1: Initial feature extraction
            # Conv2d: Convolutional layer - slides filters across the image
            # - in_channels: Number of input channels (3 for RGB)
            # - base_channels: Number of filters to learn
            # - kernel_size=5: Each filter looks at 5x5 pixel regions
            # - padding=2: Add 2 pixels of padding on each side (General rule: padding = (K-1)/2, K = kernel size)
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2),
            # GroupNorm: Normalizes features for stable training
            # - Groups features into 8 groups and normalizes within each group
            nn.GroupNorm(8, base_channels),
            # ReLU: non-linear activation function 
            # - Only keeps positive values, sets negative to 0
            # - inplace=True saves memory by modifying input directly
            nn.ReLU(inplace=True),
            
            # Layer 2: second-order feature extraction
            # - Doubles the number of channels (allows to learn more complex features)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: Final feature projection
            # - Projects to desired output dimension
            # - Spatial resolution is preserved throughout
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # === Learnable Positional Encoding ===
        # Rationale:
        # - CNNs are translation-invariant: a cat at (10,10) looks the same as at (50,50)
        # - But for binding, we NEED to know WHERE things are!
        # - Positional encoding attaches additional spatial information to the Feature Tensor
        #
        # Implementation details:
        # - 8 channels: Enough to encode X,Y positions and their combinations
        # - 64x64 default size: image size = 64
        # - Learnable: The model learns the best way to encode positions
        # - Small initialization (*0.02): Start with weak position signal, let model adjust
        #
        # Shape: (1, 8, 64, 64)
        # - 1: Batch dimension (will be expanded for all images)
        # - 8: Number of positional channels
        # - 64x64: Spatial dimensions (can be interpolated to match actual image size as needed)
        self.pos_embed = nn.Parameter(torch.randn(1, 8, 64, 64) * 0.02)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize network weights.
        
        Weight initialization is critical for neural networks:
        - Too small: Signals vanish as they pass through layers
        - Too large: Signals explode, causing instability
        - Just right: Signals maintain consistent magnitude
        
        Design choice: Kaiming initialization
        Why: it was designed specifically for ReLU networks.
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization for convolutional layers
                # - 'fan_out' mode: Considers number of output connections
                # - Helps maintain signal magnitude through the network
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    # Initialize biases to zero (standard practice)
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                # For normalization layers:
                # - Weight = 1: No scaling initially
                # - Bias = 0: No shift initially
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.
        
        This function:
        1. Extracts visual features using the CNN
        2. Adds positional information
        3. Returns combined features tensor ready for binding
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
               - B: Batch size (number of images)
               - C: Channels (3 for RGB)
               - H: Height in pixels
               - W: Width in pixels
            
        Returns:
            Features tensor of shape (B, out_channels + 8, H, W)
            - out_channels: Visual features from CNN
            - +8: Positional encoding channels
            - Spatial dimensions (H,W) are preserved
            
        Raises:
            ValueError: If input shape is invalid
        """
        # Validate inputs: check input is a 4D tensor (batch of images)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        B, C, H, W = x.shape
        
        # Check number of channels matches expectation
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")
        
        # === Step 1: Extract Visual Features ===
        # Pass image through CNN to get feature maps
        # Each location (i,j), (i ∈ H, j ∈ W) contains a feature vector describing that region
        features = self.backbone(x)  # (B, out_channels, H, W)
        
        # === Step 2: Add Positional Encoding ===
        # Resize positional embedding to match current image size (default is 64x64)
        pos_embed = F.interpolate(
            self.pos_embed,           # Our learned positional encoding
            size=(H, W),              # Target size (match input image)
            mode='bilinear',          # Smooth interpolation
            align_corners=False       # Standard setting for better results
        )
        # Expand positional encoding for all images in batch
        # -1 means "keep this dimension as is"
        pos_embed = pos_embed.expand(B, -1, -1, -1)  # (1,8,H,W) -> (B,8,H,W)
        
        # === Step 3: Combine Features and Positions ===
        # Concatenate along channel dimension (dim=1)
        # Result has both "what" (features) and "where" (positions) information
        features_with_pos = torch.cat([features, pos_embed], dim=1)
        
        return features_with_pos  # (B, out_channels + 8, H, W)