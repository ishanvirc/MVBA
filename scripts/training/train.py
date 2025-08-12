"""
 Training Script for MVBA

This script trains the MVBA model on the pre-generated dataset with per-epoch checkpointing and metrics logging
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add src to path so we can import our model modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Model import will be done dynamically based on variant
from src.losses import MVBALoss         # Custom loss function for MVBA
from src.metrics import MVBAMetrics      # Metrics computation (ARI, mIoU, etc.)
from src.visualization import MVBAVisualizer, VisualizationConfig  # For creating visual outputs

class PreGeneratedSimpleObjectsDataset(Dataset):
    """
    Dataset for loading pre-generated SimpleObjects images from disk.
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            data_dir: Base directory containing the dataset
            split: 'train' or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        
        # Verify directory exists
        if not self.split_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.split_dir}")
        
        # Load dataset info for normalization
        info_path = self.data_dir / 'dataset_info.json'
        with open(info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        # Get normalization stats (computed from entire dataset)
        # mean and std are per-channel (R, G, B)
        self.mean = torch.tensor(self.dataset_info['normalization']['mean'])
        self.std = torch.tensor(self.dataset_info['normalization']['std'])
        
        # Get list of images
        self.image_dir = self.split_dir / 'images'
        self.metadata_dir = self.split_dir / 'metadata'
        
        # Sort to ensure consistent ordering
        self.image_files = sorted(list(self.image_dir.glob('image_*.png')))
        self.metadata_files = sorted(list(self.metadata_dir.glob('metadata_*.json')))
        
        assert len(self.image_files) == len(self.metadata_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.metadata_files)} metadata"
        
        print(f"Loaded {split} dataset with {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Load image and metadata.
        
        Returns:
            image_tensor: Normalized image tensor (C, H, W)
        """
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        # Normalize: (x - mean) / std for each channel
        # unsqueeze adds dimensions for broadcasting: (C,) -> (C,1,1)
        image_tensor = (image_tensor - self.mean.unsqueeze(1).unsqueeze(2)) / self.std.unsqueeze(1).unsqueeze(2)
        
        return image_tensor


def save_checkpoint(epoch, model, optimizer, metrics, checkpoint_dir):
    """Save training checkpoint."""
    # Create checkpoint dictionary with all training state
    checkpoint = {
        'epoch': epoch,                                     # Current epoch number
        'model_state_dict': model.state_dict(),             # Model weights and parameters
        'optimizer_state_dict': optimizer.state_dict(),     # Optimizer state (momentum, etc.)
        'metrics': metrics,                                 # Training metrics for this epoch
        'timestamp': datetime.now().isoformat()             # When checkpoint was saved
    }
    
    checkpoint_path = checkpoint_dir / f'epoch_{epoch:02d}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    total_loss = 0
    # Track individual loss components for analysis
    loss_components = {'reconstruction': 0}
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, images in enumerate(progress_bar):
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear gradients from previous iteration
        output = model(images)  # Run model: images -> slots, masks, reconstruction
        
        # Compute loss (reconstruction + regularization terms)
        loss_dict = criterion(
            reconstruction=output['reconstruction'],
            target=images,
            masks=None
        )
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()     # Compute gradients via backpropagation
        optimizer.step()    # Update model parameters using gradients
        
        # Update metrics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{loss_dict['reconstruction'].item():.4f}"
        })
    
    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    return avg_loss, loss_components


def parse_args():
    parser = argparse.ArgumentParser(description='Train MVBA model variants for ablation studies')
    parser.add_argument('--model-variant', type=str, default='full',
                        choices=['baseline', 'spatial', 'feature', 'spatial_fixed', 'feature_fixed', 'full_fixed', 'full'],
                        help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # === Training Configuration ===
    DATA_DIR = '/home/ishanvir-choongh/FBNN/MVBA/data/simple_objects'  # Pre-generated dataset
    OUTPUT_DIR = f'/home/ishanvir-choongh/FBNN/MVBA/train_{args.model_variant}'  # Model-specific output
    EPOCHS = args.epochs          # Number of complete passes through dataset
    BATCH_SIZE = args.batch_size  # Images processed together
    LEARNING_RATE = 1e-4          # Adam optimizer step size
    NUM_WORKERS = 4               # Parallel data loading threads
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU if available
    SEED = 42                     # For reproducibility
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)              # PyTorch CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)     # PyTorch GPU operations
    np.random.seed(SEED)                 # NumPy operations
    
    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    checkpoint_dir = output_dir / 'checkpoints'
    vis_dir = output_dir / 'visualizations'
    metrics_dir = output_dir / 'metrics'
    
    for dir_path in [checkpoint_dir, vis_dir, metrics_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging
    log_file = output_dir / 'training_log.txt'
    
    def log_message(msg):
        """Log message to both console and file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    
    log_message(f"Starting MVBA training on pre-generated dataset")
    log_message(f"Device: {DEVICE}")
    log_message(f"Output directory: {output_dir}")
    
    # Create dataset and dataloader
    log_message("Loading dataset...")
    train_dataset = PreGeneratedSimpleObjectsDataset(DATA_DIR, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,    # Number of images per batch
        shuffle=True,             # Randomize order each epoch
        num_workers=NUM_WORKERS,  # Parallel data loading
        pin_memory=True          # Faster GPU transfer
    )
    
    # Import model based on variant (for ablation studies)
    if args.model_variant == 'baseline':
        from src.models.mvba_baseline import MVBA      # No alpha, spatial, or feature binding
    elif args.model_variant == 'spatial':
        from src.models.mvba_spatial import MVBA       # Only spatial binding (WHERE)
    elif args.model_variant == 'feature':
        from src.models.mvba_feature import MVBA       # Only feature binding (WHAT)
    elif args.model_variant == 'spatial_fixed':
        from src.models.mvba_spatial_fixed import MVBA # Spatial binding with fixed alpha=1
    elif args.model_variant == 'feature_fixed':
        from src.models.mvba_feature_fixed import MVBA # Feature binding with fixed alpha=1
    elif args.model_variant == 'full_fixed':
        from src.models.mvba_full_fixed import MVBA    # Both bindings with fixed alpha=1
    else:  # 'full'
        from src.models.mvba import MVBA               # Complete model with learned alpha
    
    # Create model
    log_message(f"Creating MVBA model variant: {args.model_variant}")
    model = MVBA(
        img_size=64,      # Input image size (64x64)
        n_slots=4,        # Number of object slots (max objects to detect)
        n_iters=3,        # Slot attention refinement iterations
        slot_dim=128,     # Dimension of slot representations
        feature_dim=64    # CNN feature dimension
    ).to(DEVICE)          # Move model to GPU if available
    
    # Create loss and optimizer
    # Loss function balances reconstruction quality with regularization
    log_message(f"Using reconstruction + entropy loss for improved slot specialization")
    criterion = MVBALoss(
        recon_weight=1.0,       # Main objective: reconstruct input
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam: adaptive learning rates
    
    # Initialize metrics and visualization
    metrics_tracker = MVBAMetrics(device=DEVICE, compute_expensive=True)  # Computes ARI, mIoU, etc.
    vis_config = VisualizationConfig(
        save_path=str(vis_dir),
        figsize=(12, 8),        # Figure size in inches
        dpi=100                 # Resolution (dots per inch)
    )
    visualizer = MVBAVisualizer(config=vis_config, device=DEVICE)  # Creates visual outputs
    
    # Training history for tracking progress
    training_history = {
        'epochs': [],           # Epoch numbers
        'train_loss': [],       # Total loss per epoch
        'loss_components': {'reconstruction': []},
        'metrics': {}           # Performance metrics
    }
    
    # Training loop
    log_message(f"\nStarting training for {EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        log_message(f"\n{'='*60}")
        log_message(f"Epoch {epoch}/{EPOCHS}")
        
        # Train
        avg_loss, loss_components = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Compute comprehensive metrics on a sample batch
        model.eval()  # Set to evaluation mode (disables dropout)
        with torch.no_grad():  # Disable gradient computation for efficiency
            # Get first batch for metrics and visualization
            sample_images = next(iter(train_loader))  # Get one batch
            sample_images = sample_images.to(DEVICE)
            sample_output = model(sample_images)  # Forward pass without gradients
            
            # Compute all metrics (reconstruction quality, slot metrics, etc.)
            metrics = metrics_tracker.compute_all_metrics(
                model_output=sample_output,    # Model predictions
                target_images=sample_images,   # Ground truth images
                target_masks=None              # No ground truth masks available
            )
        
        # Log results
        epoch_time = time.time() - epoch_start
        log_message(f"Epoch time: {epoch_time:.1f}s")
        log_message(f"Average loss: {avg_loss:.4f}")
        log_message(f"Loss components: " + 
                   ", ".join([f"{k}={v:.4f}" for k, v in loss_components.items()]))
        log_message(f"Key metrics: " +
                   f"PSNR={metrics['reconstruction_quality']['psnr']:.2f}, " +
                   f"SSIM={metrics['reconstruction_quality']['ssim']:.3f}, " +
                   f"Slot diversity={metrics['slot_utilization']['slot_diversity']:.3f}")
        
        # Save checkpoint
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics={'train_loss': avg_loss, **metrics},
            checkpoint_dir=checkpoint_dir
        )
        
        # Create visualizations for analysis
        log_message("Creating visualizations...")
        
        # Reconstruction quality - compare input vs output
        visualizer.plot_reconstruction_quality(
            original_images=sample_images[:4],      # First 4 images
            reconstructions=sample_output['reconstruction'][:4],  # Model reconstructions
            save_path=str(vis_dir / f'reconstruction_epoch_{epoch:02d}.png')
        )
        
        # Slot assignments - visualize object segmentation
        visualizer.plot_slot_assignments(
            images=sample_images[:4],                           # Original images
            masks=sample_output['masks'][:4],                   # Decoder masks
            spatial_attention=sample_output['spatial_attention'][:4],  # Attention maps
            save_path=str(vis_dir / f'slots_epoch_{epoch:02d}.png')
        )
        
        # Update training history with this epoch's results
        training_history['epochs'].append(epoch)
        training_history['train_loss'].append(avg_loss)
        for key, value in loss_components.items():
            training_history['loss_components'][key].append(value)
        
        # Store selected metrics for tracking progress
        if 'psnr' not in training_history['metrics']:
            training_history['metrics'] = {
                'psnr': [],                    # Peak Signal-to-Noise Ratio (higher=better)
                'ssim': [],                    # Structural Similarity (0-1, higher=better)
                'binding_consistency': [],      # How stable are slot assignments
                'slot_diversity': []           # How different are the slots
            }
        training_history['metrics']['psnr'].append(metrics['reconstruction_quality']['psnr'])
        training_history['metrics']['ssim'].append(metrics['reconstruction_quality']['ssim'])
        training_history['metrics']['binding_consistency'].append(metrics['binding_consistency']['spatial_consistency'])
        training_history['metrics']['slot_diversity'].append(metrics['slot_utilization']['slot_diversity'])
    
    # Training complete
    total_time = time.time() - start_time
    log_message(f"\n{'='*60}")
    log_message(f"Training complete! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Plot training curves to visualize learning progress
    log_message("Creating final training curves...")
    visualizer.plot_training_curves(
        training_logs={
            'Total Loss': training_history['train_loss'],  # Overall loss
            **{f'{k.title()} Loss': v for k, v in training_history['loss_components'].items()}  # Individual components
        },
        save_path=str(vis_dir / 'training_curves.png')
    )
    
    # Save training history as JSON for later analysis
    history_path = metrics_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)  # Pretty print with indentation
    log_message(f"Saved training history to {history_path}")
    
    # Save final model checkpoint with complete training state
    final_model_path = output_dir / 'final_model_epoch50.pth'
    torch.save({
        'epoch': EPOCHS,                                # Final epoch number
        'model_state_dict': model.state_dict(),         # All model weights
        'optimizer_state_dict': optimizer.state_dict(), # Optimizer state for resume
        'training_history': training_history             # Complete training metrics
    }, final_model_path)
    log_message(f"Saved final model to {final_model_path}")
    
    log_message("\nTraining script complete!")

# Entry point - only run main() if script is executed directly
if __name__ == '__main__':
    main()