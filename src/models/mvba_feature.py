"""
Main MVBA Model - Minimal Viable Binding Architecture

This is the core model that implements the MVBA architecture.
It combines feature extraction, slot attention, spatial binding, feature binding, and alpha generation
to create object-centric representations of images.

This model learns to:
1. Look at an image and extract meaningful features
2. Create an initial representation of objects through pure competition (slot attention)
3. Generate alpha values to control how sharply slots compete (alpha generation)
4. Understand "where" each object is in the image (spatial binding)
5. Understand "what are the properties of each object?" (feature binding)
6. Reconstruct the original image from these object representations

The binding problem: When you see a red square and a blue circle, your brain needs
to "bind" red with square and blue with circle. 
This model learns to do that!

Architecture Overview:
1. Feature extraction with positional encoding
   - CNN extracts visual features
   - Positional encoding adds location information
   
2. Slot attention for object-centric representations
   - "Slots" are like containers for objects
   - Each slot competes to represent different objects
   
3. Alpha generation from slot states
   - Alpha controls how sharply slots compete
   - Higher alpha = more winner-take-all behavior
   
4. Spatial binding (WHERE) with power-law enhancement
   - Determines which pixels belong to which object
   - Uses alpha to sharpen the competition
   
5. Feature binding (WHAT) with power-law enhancement
   - Binds visual features to the correct objects
   - Power-law makes distinctive features stand out more

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .feature_extractor import FeatureExtractor
from .slot_attention import SlotAttention
from .alpha_generator import AlphaGenerator
# Spatial binding is disabled for feature variant
from .feature_binding import FeatureBinding


class MVBA(nn.Module):
    """
    MVBA: Minimal Viable Binding Architecture
    
    This is the main model that puts everything together. It takes an image
    and produces object-centric representations that can reconstruct the image.
    """
    
    def __init__(
        self,
        # Input dimensions
        in_channels: int = 3,       # RGB
        img_size: int = 64,         # Expected input image size (assumes square, 64x64)
        
        # Architecture dimensions
        base_channels: int = 64,    # Without positional encoding
        feature_dim: int = 64,      # CNN Feature dimension
        pos_dim: int = 8,           # Positional encoding dimension
        slot_dim: int = 128,        # Slot representation dimension
        
        # Slot attention parameters
        n_slots: int = 4,           # Number of object slots (how many objects we can represent)
        n_iters: int = 3,           # Number of slot attention iterations (how many times slots refine themselves)
        n_heads: int = 4,           # Number of attention heads (multi-head attention for better representation)
        mlp_hidden_dim: int = 256,  # Hidden dimension for MLPs in slot attention
        
        # Alpha generation parameters
        min_alpha: float = 1.0,     # Minimum alpha value for sharpening
        max_alpha: float = 3.0,     # Maximum alpha value for sharpening
        
        # Binding parameters
        n_refinement_layers: int = 3,
        
        # Training parameters
        dropout: float = 0.1
    ):
        """
        Initialize MVBA model.
        
        Args:
            in_channels: Number of input image channels
            img_size: Expected input image size (assumes square)
            base_channels: Base channels for feature extractor
            feature_dim: Feature dimension (before positional encoding)
            pos_dim: Positional encoding dimension
            slot_dim: Slot representation dimension
            n_slots: Number of object slots
            n_iters: Number of slot attention iterations
            n_heads: Number of attention heads
            mlp_hidden_dim: Hidden dimension for MLPs
            min_alpha: Minimum alpha value for sharpening
            max_alpha: Maximum alpha value for sharpening
            n_refinement_layers: Number of spatial refinement layers
            dropout: Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Defensive Progremming: Validate input parameters
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be positive, got {n_iters}")
        if min_alpha >= max_alpha:
            raise ValueError(f"min_alpha must be less than max_alpha, got {min_alpha} >= {max_alpha}")
        
        # Store configuration
        self.in_channels = in_channels
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.pos_dim = pos_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.n_iters = n_iters
        
        # Total feature dimension = visual features + positional encoding
        # Positional encoding helps the model know WHERE features come from
        self.total_feature_dim = feature_dim + pos_dim
        
        # === Initialize all model components ===
        
        # 1. Feature Extractor: CNN that extracts visual + spatial features from images
        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=feature_dim
        )
        
        # 2. Slot Attention: Iterative competitive binding - slots compete for features
        # via normalized attention (minimal version of Slot Attention from Locatello et al., 2020.)
        self.slot_attention = SlotAttention(
            feature_dim=self.total_feature_dim,
            slot_dim=slot_dim,
            n_slots=n_slots,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # 3. Alpha Generator: Produces alpha values from slot states
        # Higher alpha = more aggressive competition
        self.alpha_generator = AlphaGenerator(
            slot_dim=slot_dim,
            hidden_dim=mlp_hidden_dim,
            min_alpha=min_alpha,
            max_alpha=max_alpha
        )
        
        # Feature variant: Spatial binding is disabled
        
        # 5. Feature Binding: Determines WHAT each object looks like
        # Binds visual features to the correct objects
        # Uses power-law enhancement to make distinctive features stand out
        self.feature_binding = FeatureBinding(
            feature_dim=self.total_feature_dim,
            slot_dim=slot_dim,
            n_heads=n_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )
        
        # === Decoder for reconstruction ===
        # The decoder takes slot representations and generates images
        # Start with a small spatial size and gradually upsample
        # This is more efficient than starting at full resolution
        decoder_start_size = img_size // 8          # 64x64 -> 8x8
        self.decoder_start_size = decoder_start_size
        
        # Step 1: Transform slot vector into spatial features
        # slot_dim -> mlp_hidden_dim -> spatial features
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            # Output size = channels * height * width
            nn.Linear(mlp_hidden_dim, base_channels * decoder_start_size * decoder_start_size),
            nn.ReLU(inplace=True)
        )
        
        # Step 2: Spatial broadcast decoder - gradually increases resolution
        decoder_layers = []
        current_size = decoder_start_size
        current_channels = base_channels
        
        # Build upsampling layers based on image size
        while current_size < img_size:
            # Each layer doubles the spatial resolution (2x upsampling)
            next_channels = max(16, current_channels // 2)  # Reduce channels as we upsample
            decoder_layers.extend([
                # ConvTranspose2d is like "reverse convolution:" increases spatial size
                nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                # GroupNorm helps with training stability
                nn.GroupNorm(min(8, next_channels // 2), next_channels),
                nn.ReLU(inplace=True)
            ])
            current_channels = next_channels
            current_size *= 2
        
        # Final layer: Generate both image (RGB) and mask (1 channel)
        # Total output channels = in_channels (RGB) + 1 (mask)
        decoder_layers.append(nn.Conv2d(current_channels, in_channels + 1, kernel_size=3, padding=1))
        
        self.spatial_decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for decoder.
        
        We use Xavier initialization which helps maintain gradient magnitudes across layers.
        """
        # Initialize the slot-to-spatial decoder
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Start biases at zero
                    nn.init.constant_(module.bias, 0)
        
        # Initialize the spatial upsampling decoder
        for module in self.spatial_decoder.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MVBA model.
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            Dictionary containing:
            - 'reconstruction': Reconstructed images (B, C, H, W)
            - 'masks': Object masks (B, n_slots, H, W)
            - 'slots': Final slot representations (B, n_slots, slot_dim)
            - 'spatial_attention': Spatial binding maps (B, n_slots, H, W)
            - 'bound_features': Bound feature representations (B, n_slots, slot_dim)
            - 'alphas': Alpha values used for sharpening
            
            
        Raises:
            ValueError: If input shape is invalid
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {images.dim()}D")
        
        B, C, H, W = images.shape
        
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"Expected {self.img_size}x{self.img_size} images, got {H}x{W}")
        
        # === Step 1: Feature Extraction ===
        # Extract visual + spatial features from the image using CNN
        features = self.feature_extractor(images)  # (B, C_feat, H, W)
        
        # === Step 2: Slot Attention (Iterative Refinement) ===
        # Initialize slots from internal learned distribution, 
        # Run multiple iterations to refine slot assignments
        slots = None
        for _ in range(self.n_iters):
            # Each iteration:
            # 1. Slots query the features: "What belongs to me?"
            # 2. Features respond based on similarity to slot queries
            # 3. Slots update based on the features they attended to
            slots, attention_weights = self.slot_attention(features, slots, return_attention=True)
        # After iterations:
        # slots: (B, n_slots, slot_dim) - refined object representations
        # attention_weights: (B, n_slots, H, W) - initial attention maps
        
        # === Step 3: Generate Alpha Values ===
        # Alpha controls competition sharpness for each slot
        # Different slots can have different alpha values based on their content
        alphas = self.alpha_generator(slots)
        # alphas: Dictionary with:
        #   'spatial': (B, n_slots, 1) - for spatial competition
        #   'feature': (B, n_slots, 1) - for feature enhancement
        
        # === Feature Variant: Skip Step 4 (Spatial Binding) ===
        # Create uniform spatial attention
        B = images.shape[0]
        H, W = features.shape[2:]
        # Each slot has equal attention (1/n_slots) across the entire image (0.25, 0.25, 0.25, 0.25)
        spatial_attention = torch.ones(B, self.n_slots, H, W, device=features.device) / self.n_slots
        
        # === Step 5: Feature Binding (WHAT) ===
        # Bind visual features to objects based on spatial attention
        # Uses alpha to enhance strong features
        bound_features, enhanced_features = self.feature_binding(
            features, spatial_attention, alphas, slots
        )
        # bound_features: (B, n_slots, slot_dim) - object feature representations
        # enhanced_features: (B, n_slots, C_feat, H, W) - enhanced feature maps
        
        # === Step 6: Decode and Reconstruct ===
        # Each slot generates its own image reconstruction
        # Final image = weighted combination of all slot reconstructions
        
        reconstructions = []  # Will store each slot's image
        masks = []            # Will store each slot's contribution mask
        
        for slot_idx in range(self.n_slots):
            # Decode this slot's features to an image
            
            # Step 6a: Transform slot vector to spatial features
            # bound_features contains what each slot "knows" about its object
            slot_features = self.decoder(bound_features[:, slot_idx])  # (B, C*H*W)
            # Reshape from flat vector to spatial tensor
            slot_features = slot_features.view(B, -1, self.decoder_start_size, self.decoder_start_size)
            # Now: (B, base_channels, 8, 8) for 64x64 images
            
            # Step 6b: Upsample to full image size
            # spatial_decoder gradually increases resolution: 8x8 -> 16x16 -> 32x32 -> 64x64
            slot_decoded = self.spatial_decoder(slot_features)  # (B, C+1, H, W)
            
            # Step 6c: Split output into image and mask
            # The decoder outputs both:
            # - RGB image: what this slot wants to draw
            # - Mask: where this slot wants to draw (importance map)
            slot_recon = slot_decoded[:, :C]      # RGB channels (B, C, H, W)
            slot_mask = slot_decoded[:, C:C+1]    # Mask channel (B, 1, H, W)
            
            reconstructions.append(slot_recon)
            masks.append(slot_mask)
        
        # Stack all slot outputs
        reconstructions = torch.stack(reconstructions, dim=1)  # (B, n_slots, C, H, W)
        masks = torch.stack(masks, dim=1).squeeze(2)  # (B, n_slots, H, W)
        
        # Normalize masks using softmax
        # This ensures masks sum to 1 at each pixel location
        # Each pixel in the final image comes from a weighted combination of slots
        masks = F.softmax(masks, dim=1)
        
        # Combine all slot reconstructions using normalized masks
        # This is like alpha blending in computer graphics
        masks_expanded = masks.unsqueeze(2)  # Add channel dimension: (B, n_slots, 1, H, W)
        # Weighted sum: each slot's image × its mask, then sum across slots
        reconstruction = (reconstructions * masks_expanded).sum(dim=1)  # (B, C, H, W)
        
        # === Step 7: Compute Metrics ===
        
        # === Prepare Output Dictionary ===
        output = {
            # Main outputs
            'reconstruction': reconstruction,          # Final reconstructed image
            'masks': masks,                            # Object segmentation masks
            'slots': slots,                            # Object representations
            'spatial_attention': spatial_attention,    # Where each object is
            'bound_features': bound_features,          # What each object looks like
            'alphas': alphas                           # Competition strengths
        }
        
        
        return output
