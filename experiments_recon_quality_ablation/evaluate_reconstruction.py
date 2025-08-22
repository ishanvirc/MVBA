"""
Reconstruction Quality Ablation - Evaluation Script

This script evaluates reconstruction quality across MVBA model variants to test
if power-law enhancements improve object binding and reconstruction.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import importlib

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.metrics import MVBAMetrics
from src.visualization import MVBAVisualizer, VisualizationConfig


class SimpleObjectsTestDataset(Dataset):
    """Dataset for loading SimpleObjects test images."""
    
    def __init__(self, data_dir: str, dataset_info_path: str, indices=None):
        """
        Initialize test dataset.
        
        Args:
            data_dir: Path to test data directory
            dataset_info_path: Path to dataset_info.json
            indices: Optional list of specific indices to use
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.metadata_dir = self.data_dir / 'metadata'
        
        # Load normalization stats
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        self.mean = torch.tensor(dataset_info['normalization']['mean'])
        self.std = torch.tensor(dataset_info['normalization']['std'])
        
        # Get image list
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(list(self.image_dir.glob('*.png')))))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # Load image
        image_path = self.image_dir / f'image_{actual_idx:06d}.png'
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean.unsqueeze(1).unsqueeze(2)) / self.std.unsqueeze(1).unsqueeze(2)
        
        # Load metadata
        metadata_path = self.metadata_dir / f'metadata_{actual_idx:06d}.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'image': image_tensor,
            'index': actual_idx
        }


def load_model_variant(config, variant_name):
    """Load a specific model variant from checkpoint."""
    variant_config = config['models']['variants'][variant_name]
    checkpoint_path = Path(config['models']['base_path']) / variant_config['checkpoint']
    
    # Import the model class
    module_path = variant_config['module']
    module = importlib.import_module(module_path)
    MVBA = module.MVBA
    
    # Create model
    model = MVBA(
        img_size=config['models']['image_size'],
        n_slots=config['models']['n_slots'],
        n_iters=config['models']['n_iters'],
        slot_dim=config['models']['slot_dim'],
        feature_dim=config['models']['feature_dim']
    )
    
    # Load checkpoint - find latest if specified doesn't exist
    if not checkpoint_path.exists():
        # Try to find latest checkpoint
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.exists():
            checkpoints = sorted(list(checkpoint_dir.glob('epoch_*.pth')))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"Using latest checkpoint for {variant_name}: {checkpoint_path.name}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {variant_name} model from {checkpoint_path}")
    
    return model


def evaluate_model(model, dataloader, metrics_computer, device, variant_name, save_dir, config):
    """Evaluate a single model variant on test data."""
    model.to(device)
    model.eval()
    
    all_metrics = []
    all_outputs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {variant_name}")):
            images = batch['image'].to(device)
            indices = batch['index']
            
            # Forward pass
            output = model(images)
            
            # Compute metrics
            batch_metrics = metrics_computer.compute_reconstruction_quality(
                reconstruction=output['reconstruction'],
                target=images,
                return_individual=True
            )
            
            # Store results
            for i in range(len(images)):
                sample_metrics = {
                    'index': indices[i].item(),
                    'mse': batch_metrics['mse'][i].item(),
                    'psnr': batch_metrics['psnr'][i].item(),
                    'ssim': batch_metrics['ssim'][i].item() if 'ssim' in batch_metrics else 0.0,
                    'l1': batch_metrics['l1'][i].item()
                }
                all_metrics.append(sample_metrics)
                
                # Save reconstruction for visualization
                all_outputs.append({
                    'index': indices[i].item(),
                    'reconstruction': output['reconstruction'][i].cpu(),
                    'masks': output['masks'][i].cpu(),
                    'spatial_attention': output['spatial_attention'][i].cpu()
                })
    
    # Save raw outputs
    output_dir = save_dir / 'raw' / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save outputs for visualization samples from config
    vis_indices = config['evaluation']['sample_indices']
    for output_data in all_outputs:
        if output_data['index'] in vis_indices or len(os.listdir(output_dir)) < 10:
            torch.save(output_data, output_dir / f"output_{output_data['index']:06d}.pt")
    
    return all_metrics, all_outputs


def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction quality across model variants')
    parser.add_argument('--config', type=str, 
                        default='../config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['evaluation']['random_seed'])
    np.random.seed(config['evaluation']['random_seed'])
    
    # Setup device
    device = torch.device(config['evaluation']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(__file__).parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    
    # Create dataset
    print("Loading test dataset...")
    # Select subset of test samples
    n_samples = config['evaluation']['n_test_samples']
    indices = np.random.choice(config['data']['total_test_samples'], n_samples, replace=False)
    indices = sorted(indices.tolist())
    
    dataset = SimpleObjectsTestDataset(
        data_dir=config['data']['test_data_path'],
        dataset_info_path=config['data']['dataset_info_path'],
        indices=indices
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize metrics computer
    metrics_computer = MVBAMetrics(
        device=device,
        compute_expensive=config['metrics']['compute_expensive']
    )
    
    # Evaluate each model variant
    all_results = {}
    
    for variant_name in config['models']['variants'].keys():
        print(f"\n{'='*60}")
        print(f"Evaluating {variant_name} model...")
        print(f"{'='*60}")
        
        # Load model
        try:
            model = load_model_variant(config, variant_name)
        except FileNotFoundError as e:
            print(f"Skipping {variant_name}: {e}")
            continue
        
        # Evaluate
        metrics, outputs = evaluate_model(
            model, dataloader, metrics_computer, device, variant_name, save_dir, config
        )
        
        all_results[variant_name] = {
            'metrics': metrics,
            'aggregate': {
                'mse': np.mean([m['mse'] for m in metrics]),
                'mse_std': np.std([m['mse'] for m in metrics]),
                'psnr': np.mean([m['psnr'] for m in metrics]),
                'psnr_std': np.std([m['psnr'] for m in metrics]),
                'ssim': np.mean([m['ssim'] for m in metrics]),
                'ssim_std': np.std([m['ssim'] for m in metrics]),
                'l1': np.mean([m['l1'] for m in metrics]),
                'l1_std': np.std([m['l1'] for m in metrics])
            }
        }
        
        # Clean up memory
        del model
        torch.cuda.empty_cache()
    
    # Save all metrics
    metrics_dir = save_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    # Save per-sample metrics
    with open(metrics_dir / 'per_sample_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregate metrics
    aggregate_results = {
        variant: results['aggregate'] 
        for variant, results in all_results.items()
    }
    with open(metrics_dir / 'aggregate_metrics.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for variant, results in aggregate_results.items():
        print(f"\n{variant.upper()}:")
        print(f"  MSE:  {results['mse']:.6f} ± {results['mse_std']:.6f}")
        print(f"  PSNR: {results['psnr']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"  SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
        print(f"  L1:   {results['l1']:.6f} ± {results['l1_std']:.6f}")
    
    # Save evaluation info
    eval_info = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'test_indices': indices,
        'n_samples': len(indices)
    }
    with open(save_dir / 'evaluation_info.json', 'w') as f:
        json.dump(eval_info, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()