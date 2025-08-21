"""
Slot Attention Module for MVBA

Reference: This is a minimal implementation of the Slot Attention Model developed by Locatello et al., 2020.

This module creates slots: a slot is a learnable, abstract representation of a portion of the input data (images in our case).
- Think of it as a container that holds information about a specific part of the input, potentially representing an object or a concept. 
- This is inspired by how the brain might have "object files" - mental representations that track individual objects in a scene.

Imagine you have a scene with a red car and blue ball:
- Slot 1 learns to attend to the car-like features
- Slot 2 learns to attend to the ball-like features
- Competition ensures each feature goes to the most appropriate slot

Critical aspects:
1. Symmetry breaking: Random initialization ensures slots start with distinct representations
2. Competitive attention: Softmax normalization across slots, not spatial locations, enforces exclusive feature assignment
3. Iterative refinement: Iterative updates via GRU-based updates allow slots to refine their representations
4. Multi-head attention: Parallel attention heads (4) provide slots with diverse perspectives (4) on the input features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SlotAttention(nn.Module):
    """
    Slot Attention: Learning object-centric representations through competition.
    
    This module implements the slot competition to represent different objects in the scene. 
    
    Key Features:
    
    1. Built-in symmetry breaking:
       - Problem: If all slots start the same, they tend to learn the same thing (uniformity)
       - Solution: Initialize each slot with different biases
       - Result: Each slot naturally specializes in different objects
    
    2. Competitive attention:
       - Instead of slots sharing features, they compete for them
       - Uses softmax across slots (not spatial locations)
    
    3. GRU-based updates:
       - GRU (Gated Recurrent Unit) allows slots to maintain "memory"
       - Slots can refine their representation through iterative improvement over multiple passes
    
    4. Multi-head attention:
       - Provides multiple perspectives on the same data
       - Each head can focus on different aspects (color, shape, etc.)
    """
    
    def __init__(
        self, 
        n_slots: int = 4, 
        slot_dim: int = 128, 
        feature_dim: int = 72,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Slot Attention module.
        
        Args:
            n_slots: Number of object slots
            slot_dim: Dimension of each slot
            feature_dim: Dimension of input features
            n_heads: Number of attention heads
            dropout: Dropout rate
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validate inputs (defensive programming)
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")
        if slot_dim <= 0 or slot_dim % n_heads != 0:
            raise ValueError(f"slot_dim must be positive and divisible by n_heads, got {slot_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = slot_dim // n_heads
        
        # === Slot Initialization Parameters ===
        # Slots are initialized from learnable Gaussian distributions
        # Each slot has its own mean (μ) and standard deviation (σ)
        
        # slot_mu: The "average" initialization for each slot
        # Shape: (1, n_slots, slot_dim) - different mean for each slot
        self.slot_mu = nn.Parameter(torch.zeros(1, n_slots, slot_dim))
        
        # slot_sigma: How much random variation to add
        # Smaller values = more consistent initialization
        self.slot_sigma = nn.Parameter(torch.ones(1, n_slots, slot_dim) * 0.1)
        
        # === Symmetry Breaking Initialization ===
        # Without this, all slots would learn the same thing
        with torch.no_grad():  # Don't track gradients during initialization
            
            for i in range(n_slots):
                # Step 1: Give each slot a unique random pattern
                # Different seed = different random pattern
                torch.manual_seed(42 + i * 100)  # Reproducible but unique
                self.slot_mu.data[0, i] = torch.randn(slot_dim) * 0.1
                
                # Step 2: Add systematic bias to different dimensions
                # This lets slots have "preferences" for different features
                # Example with 4 slots and 128 dims:
                # - Slot 0: Strong in dims 0-31
                # - Slot 1: Strong in dims 32-63
                # - Slot 2: Strong in dims 64-95
                # - Slot 3: Strong in dims 96-127
                chunk_size = slot_dim // n_slots
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, slot_dim)
                self.slot_mu.data[0, i, start_idx:end_idx] += 0.5  # Strong bias
        
        # === Attention Components (Query-Key-Value) ===
        # These implement the attention mechanism where slots "query" features
        
        # to_q: Transforms slot state into a "query" - "What am I looking for?"
        self.to_q = nn.Linear(slot_dim, slot_dim)
        
        # to_k: Transforms features into "keys" - "What information is here?"
        self.to_k = nn.Linear(feature_dim, slot_dim)
        
        # to_v: Transforms features into "values" - "Here's the actual content"
        self.to_v = nn.Linear(feature_dim, slot_dim)
        
        # === Slot Update Components ===
        # GRU (Gated Recurrent Unit): Updates slots based on attended features
        # Gates control information flow: update gate modulates new content, reset gate controls previous state influence
        # GRUCell: Processes each slot independently
        # - slot_dim: Input and output dimension of the GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # Layer normalization: Keeps slot values stable
        self.norm_pre = nn.LayerNorm(slot_dim)   # Before attention
        self.norm_post = nn.LayerNorm(slot_dim)  # Before MLP
        
        # MLP for slot refinement: Further processes the updated slots
        # Expansion to 2x dimensions allows learning complex transformations
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),    # Expand
            nn.ReLU(inplace=True),                # Non-linearity
            nn.Dropout(dropout),                  # Regularization
            nn.Linear(slot_dim * 2, slot_dim),    # Contract
            nn.Dropout(dropout)                   # Regularization
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for linear layers and GRU."""
        # Linear layers
        for module in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        
        # GRU initialization
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.constant_(self.gru.bias_ih, 0)
        nn.init.constant_(self.gru.bias_hh, 0)
        
        # MLP initialization
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def initialize_slots(self, 
                        batch_size: int, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """ 
        Initialize slots with built-in diversity to break symmetry.
        
        Critical for specialization: Samples form learned Gaussian distributions 
        with dimensional biases to ensure each slot starts with a unique representation.
        
        The initialization strategy:
        1. Start from learned distributions (mu, sigma for each slot)
        2. Add random noise (different for each image in batch)
        3. Add deterministic biases in different dimensions. 
        
        Args:
            batch_size: Number of samples in batch
            device: Device to create slots on (CPU/GPU)
            
        Returns:
            Initialized slots of shape (batch_size, n_slots, slot_dim)
            Each slot is uniquely initialized to prevent symmetry
        """
        if device is None:
            device = self.slot_mu.device
            
        # Step 1: Expand learned parameters to batch size
        # mu: mean initialization for each slot
        # sigma: standard deviation for each slot
        mu = self.slot_mu.expand(batch_size, -1, -1)      # (1, n_slots, slot_dim) -> (B, n_slots, slot_dim)
        sigma = self.slot_sigma.expand(batch_size, -1, -1)
        
        # Step 2: Sample from Gaussian distribution
        # Different random noise for each image ensures diversity across batch
        noise = torch.randn(batch_size, self.n_slots, self.slot_dim, device=device)
        slots = mu + sigma * noise  # Gaussian sampling: N(mu, sigma)
        
        # Step 3: Additional deterministic symmetry breaking in the first 16 dimensions
        # Assign non-overlapping dimensional subspaces to each slot
        # - Gives each slot a "preference" for different dimensions
        # - Encourages specialization through activation biasing
        for i in range(self.n_slots):
            # Create spatial bias for slot i
            spatial_bias = torch.zeros_like(slots[:, i])  # (batch_size, slot_dim)
            
            # Give each slot enhanced activation in different dimensions
            # With 4 slots, slot_dim=128:
            # - Slot 0: Boosts dims [0:4]     (indices 0,1,2,3)
            # - Slot 1: Boosts dims [4:8]     (indices 4,5,6,7)  
            # - Slot 2: Boosts dims [8:12]    (indices 8,9,10,11)
            # - Slot 3: Boosts dims [12:16]   (indices 12,13,14,15)
            spatial_bias[:, i*4:(i+1)*4] += 0.3  # Add 0.3 to these specific dimensions
            
            # Apply dimensional bias
            slots[:, i] += spatial_bias
            
        return slots
    
    def forward(
        self, 
        features: torch.Tensor, 
        slots: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Update slots based on features through competitive attention.
        
        Args:
            features: Input features of shape (B, C, H, W)
            slots: (B, n_slots, slot_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Updated slots of shape (B, n_slots, slot_dim)
            - Optional attention weights of shape (B, n_slots, H, W) if requested
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # Validate inputs: check features is a 4D tensor (batch of images)
        if features.dim() != 4:
            raise ValueError(f"Expected 4D features tensor, got {features.dim()}D")
        
        B, C, H, W = features.shape
        # Check number of channels matches expectation
        if C != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature channels, got {C}")
        
        # Initialize slots if not provided
        if slots is None:
            slots = self.initialize_slots(B, device=features.device)
        else:
            if slots.shape != (B, self.n_slots, self.slot_dim):
                raise ValueError(f"Expected slots shape {(B, self.n_slots, self.slot_dim)}, got {slots.shape}")
        
        # === Step 1: Prepare Features ===
        # Flatten spatial dimensions for attention computation
        # (B, C, H, W) -> (B, H*W, C)
        # This treats each spatial location as a separate "token"
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # === Step 2: Generate Queries, Keys, Values ===
        
        # Normalize slots before generating queries (improves stability)
        slots_normed = self.norm_pre(slots)
        
        # Queries: "What is each slot looking for?"
        q = self.to_q(slots_normed)  # (B, n_slots, slot_dim)
        
        # Keys: "What information is available at each location?"
        k = self.to_k(features_flat)  # (B, H*W, slot_dim)
        
        # Values: "The actual feature content to aggregate"
        v = self.to_v(features_flat)  # (B, H*W, slot_dim)
        
        # === Step 3: Reshape for Multi-Head Attention ===
        # Multi-head attention allows the model to jointly attend to information
        # from different representation subspaces at different positions.
        # Each slot's query is split into multiple "heads" to capture diverse features
        # - Each head can focus on different aspects of the input
        # - Head 1 might focus on feature A
        # - Head 2 might focus on feature B
        # - Head 3 might focus on feature C
        # - Head 4 might focus on feature D
        
        # Split each slot/feature into multiple "heads"
        # [slot with 128 dims] -> Split into: [head1:32d, head2:32d, head3:32d, head4:32d]
        
        # Reshape queries: (B, n_slots, slot_dim) -> (B, n_slots, n_heads, head_dim)
        # Each slot's query is split into n_heads smaller queries
        q = q.view(B, self.n_slots, self.n_heads, self.head_dim)
        
        # Reshape keys: (B, H*W, slot_dim) -> (B, H*W, n_heads, head_dim)  
        # Each spatial location's key is split into n_heads smaller keys
        k = k.view(B, H*W, self.n_heads, self.head_dim)
        
        # Reshape values: same transformation as keys
        # Each spatial location's value is split into n_heads smaller values
        v = v.view(B, H*W, self.n_heads, self.head_dim)
        
        # Transpose for efficient batch computation
        # PyTorch's matmul expects: (batch, ..., rows, cols)
        # We want to compute attention for all heads in parallel
        # After transpose, dim 1 = heads, allowing parallel head computation (matmul is performed on dim 2 and 3)
        q = q.transpose(1, 2)  # (B, n_heads, n_slots, head_dim)
        k = k.transpose(1, 2)  # (B, n_heads, H*W, head_dim)
        v = v.transpose(1, 2)  # (B, n_heads, H*W, head_dim)
        
        # === Step 4: Compute Attention Scores ===
        # For each head, compute how well each slot's query matches each position's key
        # High score = "This slot's query matches this position's key well"
        
        # Scale by 1/sqrt(d)
        # Without scaling, dot products can become very large for high dimensions,
        # causing softmax to output extreme values (near 0 or 1), leading to vanishing gradients
        scale = self.head_dim ** -0.5  # 1/sqrt(head_dim) keeps values in reasonable range
        
        # Compute similarity scores between all query-key pairs
        # k.transpose(-2, -1): swap last two dims to get (B, n_heads, head_dim, H*W)
        # Result: for each head, a score matrix of (n_slots x H*W)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, n_slots, H*W)
        
        # === Step 5: Apply Competitive Normalization ===
        # This is the key difference in slot attention from standard attention. 
        # Instead of normalizing across spatial locations (dim=3), normalize across SLOTS (dim=2)
        # This creates competition between slots for the same spatial location.
        
        # Apply softmax directly on dim=2 (slots)
        # At each spatial location, slots compete for assignment
        attn = F.softmax(attn, dim=2)  # (B, n_heads, n_slots, H*W)
        
        # === Step 6: Aggregate Features ===
        # Apply attention weights to values to get weighted features from all locations
        aggregated = torch.matmul(attn, v)  # (B, n_heads, n_slots, head_dim)
        
        # === Step 7: Reshape Back to Original Format ===
        # Combine multi-head outputs back into single representation
        aggregated = aggregated.transpose(1, 2)                         # (B, n_slots, n_heads, head_dim)
        aggregated = aggregated.reshape(B, self.n_slots, self.slot_dim) # (B, n_slots, slot_dim)
        
        # === Step 8: Update Slots via GRU ===
        # GRU (Gated Recurrent Unit) intelligently updates slot states
        # It decides what to keep from old state and what to update from new info
        # Flatten for GRU processing (GRU expects 2D input)
        slots_flat = slots.reshape(B * self.n_slots, self.slot_dim)
        aggregated_flat = aggregated.reshape(B * self.n_slots, self.slot_dim)
        
        # GRU update: new_state = GRU(input, hidden_state)
        # - input: aggregated features from attention
        # - hidden_state: current slot state
        # Reset gate: r = σ(W_r·[input, hidden])  - controls how much past to forget
        # Update gate: z = σ(W_z·[input, hidden]) - controls how much to update
        # Candidate: h̃ = tanh(W·[input, r⊙hidden]) - new content modulated by reset
        # Output: h' = (1-z)⊙hidden + z⊙h̃ - interpolate old and new based on update gate
        updated_slots = self.gru(aggregated_flat, slots_flat)
        # Reshape back to (B, n_slots, slot_dim)
        updated_slots = updated_slots.reshape(B, self.n_slots, self.slot_dim)
        
        # === Step 9: Apply MLP Refinement ===
        # Two-layer feedforward network with residual connection
        # LayerNorm → Linear(128→256) → ReLU → Dropout → Linear(256→128) → Dropout
        refined = self.mlp(self.norm_post(updated_slots))
        # Residual connection: preserves gradient flow and enables learning incremental updates
        # Without residual: forces MLP to learn full representation; With residual: MLP learns delta
        updated_slots = updated_slots + refined
        
        # === Optional: Return Attention Maps ===
        if return_attention:
            # Average attention across all heads for visualization
            attn_avg = attn.mean(dim=1)  # (B, n_slots, H*W)
            # Reshape back to spatial format
            attn_reshaped = attn_avg.reshape(B, self.n_slots, H, W)
            return updated_slots, attn_reshaped
        
        return updated_slots, None
