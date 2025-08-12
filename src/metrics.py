"""
Comprehensive Validation Metrics for MVBA Architecture

This module provides metrics for evaluating the quality of object binding in the
Minimal Viable Binding Architecture (MVBA). The metrics cover all aspects of
binding quality from spatial consistency to reconstruction fidelity.

Key metric categories:
1. Binding Consistency - How well objects are bound across features/spatial locations
2. Slot Utilization - Efficiency and specialization of slot usage
3. Segmentation Quality - IoU, ARI, and mask quality metrics
4. Alpha Statistics - Analysis of alpha value distributions and stability
5. Reconstruction Quality - PSNR, SSIM, and perceptual metrics
6. Binding Interpretability - Attention clarity and feature coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, some metrics will use fallback implementations")

try:
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, ARI will use simplified implementation")


class MVBAMetrics:
    """
    Comprehensive metrics suite for evaluating MVBA model performance.
    
    This class provides both individual and aggregate statistics for all aspects
    of object binding quality. Metrics are designed to be computationally efficient
    for use during training while providing detailed analysis capabilities.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        eps: float = 1e-8,
        compute_expensive: bool = True
    ):
        """
        Initialize MVBA metrics calculator.
        
        Args:
            device: Device to use for computations (defaults to input tensor device)
            eps: Small epsilon for numerical stability
            compute_expensive: Whether to compute expensive metrics (like SSIM)
        """
        self.device = device
        self.eps = eps
        self.compute_expensive = compute_expensive
        
        # Pre-computed constants for efficiency
        self._sqrt2 = np.sqrt(2.0)
        self._log2 = np.log(2.0)
    
    def _linear_sum_assignment_fallback(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback implementation for linear sum assignment (greedy)."""
        cost = cost_matrix.copy()
        n_rows, n_cols = cost.shape
        row_ind = []
        col_ind = []
        
        for _ in range(min(n_rows, n_cols)):
            # Find minimum cost assignment
            min_idx = np.unravel_index(np.argmin(cost), cost.shape)
            row_ind.append(min_idx[0])
            col_ind.append(min_idx[1])
            
            # Remove assigned row and column
            cost[min_idx[0], :] = np.inf
            cost[:, min_idx[1]] = np.inf
        
        return np.array(row_ind), np.array(col_ind)
    
    def _adjusted_rand_score_fallback(self, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        """Simplified fallback implementation for adjusted rand index."""
        # Count agreements and disagreements
        n = len(labels_true)
        if n <= 1:
            return 1.0
            
        # Count pairs with same/different labels
        agreements = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                same_true = (labels_true[i] == labels_true[j])
                same_pred = (labels_pred[i] == labels_pred[j])
                
                if same_true == same_pred:
                    agreements += 1
                total_pairs += 1
        
        if total_pairs == 0:
            return 1.0
            
        # Simple accuracy-based score (not exactly ARI but similar interpretation)
        return agreements / total_pairs
    
    def compute_all_metrics(
        self,
        model_output: Dict[str, torch.Tensor],
        target_images: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor, Dict[str, float]]]:
        """
        Compute all available metrics for MVBA model output.
        
        Args:
            model_output: Dictionary from MVBA forward pass containing:
                - 'reconstruction': Reconstructed images (B, C, H, W)
                - 'masks': Object masks (B, n_slots, H, W)
                - 'slots': Slot representations (B, n_slots, slot_dim)
                - 'spatial_attention': Spatial binding maps (B, n_slots, H, W)
                - 'bound_features': Bound feature representations (B, n_slots, slot_dim)
                - 'alphas': Alpha values dict with 'spatial' and 'feature' keys
                - 'binding_entropy': Entropy of spatial binding (B,)
                - 'feature_diversity': Diversity of bound features (B,)
            target_images: Ground truth images (B, C, H, W)
            target_masks: Optional ground truth masks (B, n_objects, H, W)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary containing all computed metrics, organized by category
        """
        metrics = {}
        
        # 1. Binding Consistency Metrics
        binding_metrics = self.compute_binding_consistency(
            model_output['spatial_attention'],
            model_output['bound_features'],
            model_output.get('enhanced_features'),
            return_individual=return_individual
        )
        metrics['binding_consistency'] = binding_metrics
        
        # 2. Slot Utilization Metrics
        slot_metrics = self.compute_slot_utilization(
            model_output['slots'],
            model_output['spatial_attention'],
            model_output['bound_features'],
            return_individual=return_individual
        )
        metrics['slot_utilization'] = slot_metrics
        
        # 3. Segmentation Quality Metrics (if ground truth available)
        if target_masks is not None:
            seg_metrics = self.compute_segmentation_quality(
                model_output['masks'],
                target_masks,
                return_individual=return_individual
            )
            metrics['segmentation_quality'] = seg_metrics
        
        # 4. Alpha Statistics
        alpha_metrics = self.compute_alpha_statistics(
            model_output['alphas'],
            model_output['slots'],
            return_individual=return_individual
        )
        metrics['alpha_statistics'] = alpha_metrics
        
        # 5. Reconstruction Quality Metrics
        recon_metrics = self.compute_reconstruction_quality(
            model_output['reconstruction'],
            target_images,
            return_individual=return_individual
        )
        metrics['reconstruction_quality'] = recon_metrics
        
        # 6. Binding Interpretability Metrics
        interp_metrics = self.compute_binding_interpretability(
            model_output['spatial_attention'],
            model_output['masks'],
            model_output['bound_features'],
            return_individual=return_individual
        )
        metrics['binding_interpretability'] = interp_metrics
        
        return metrics
    
    def compute_binding_consistency(
        self,
        spatial_attention: torch.Tensor,
        bound_features: torch.Tensor,
        enhanced_features: Optional[torch.Tensor] = None,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Compute binding consistency metrics.
        
        Args:
            spatial_attention: Spatial binding maps (B, n_slots, H, W)
            bound_features: Bound feature representations (B, n_slots, slot_dim)
            enhanced_features: Enhanced features (B, n_slots, C_feat, H, W)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary with binding consistency metrics
        """
        B, n_slots, H, W = spatial_attention.shape
        device = spatial_attention.device
        
        metrics = {}
        
        # Object-level binding consistency
        # Measure how consistently each object is bound across spatial locations
        spatial_consistency = self._compute_spatial_consistency(spatial_attention)
        metrics['spatial_consistency'] = self._aggregate_metric(
            spatial_consistency, return_individual
        )
        
        # Cross-modal binding consistency
        # Compare spatial attention with feature-based similarities
        if bound_features is not None:
            cross_modal_consistency = self._compute_cross_modal_consistency(
                spatial_attention, bound_features
            )
            metrics['cross_modal_consistency'] = self._aggregate_metric(
                cross_modal_consistency, return_individual
            )
        
        # Temporal consistency (for sequence data)
        # Note: This would require sequence inputs, placeholder for now
        metrics['temporal_consistency'] = 0.0
        
        # Binding stability - measure how stable bindings are across iterations
        binding_stability = self._compute_binding_stability(spatial_attention)
        metrics['binding_stability'] = self._aggregate_metric(
            binding_stability, return_individual
        )
        
        return metrics
    
    def compute_slot_utilization(
        self,
        slots: torch.Tensor,
        spatial_attention: torch.Tensor,
        bound_features: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Compute slot utilization metrics.
        
        Args:
            slots: Slot representations (B, n_slots, slot_dim)
            spatial_attention: Spatial binding maps (B, n_slots, H, W)
            bound_features: Bound features (B, n_slots, slot_dim)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary with slot utilization metrics
        """
        B, n_slots, slot_dim = slots.shape
        
        metrics = {}
        
        # Slot usage efficiency
        usage_efficiency = self._compute_slot_usage_efficiency(spatial_attention)
        metrics['usage_efficiency'] = self._aggregate_metric(
            usage_efficiency, return_individual
        )
        
        # Slot specialization
        specialization = self._compute_slot_specialization(bound_features)
        metrics['specialization'] = self._aggregate_metric(
            specialization, return_individual
        )
        
        # Load balancing
        load_balance = self._compute_load_balancing(spatial_attention)
        metrics['load_balance'] = self._aggregate_metric(
            load_balance, return_individual
        )
        
        # Slot diversity
        slot_diversity = self._compute_slot_diversity(slots)
        metrics['slot_diversity'] = self._aggregate_metric(
            slot_diversity, return_individual
        )
        
        return metrics
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Compute segmentation quality metrics.
        
        Args:
            pred_masks: Predicted masks (B, n_slots, H, W)
            target_masks: Ground truth masks (B, n_objects, H, W)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary with segmentation quality metrics
        """
        metrics = {}
        
        # IoU with optimal assignment
        iou_scores = self._compute_optimal_iou(pred_masks, target_masks)
        metrics['mean_iou'] = self._aggregate_metric(iou_scores, return_individual)
        
        # Adjusted Rand Index
        ari_scores = self._compute_ari_scores(pred_masks, target_masks)
        metrics['ari'] = self._aggregate_metric(ari_scores, return_individual)
        
        # Mask coverage and completeness
        coverage, completeness = self._compute_mask_coverage_completeness(
            pred_masks, target_masks
        )
        metrics['coverage'] = self._aggregate_metric(coverage, return_individual)
        metrics['completeness'] = self._aggregate_metric(completeness, return_individual)
        
        # Boundary accuracy
        boundary_f1 = self._compute_boundary_f1(pred_masks, target_masks)
        metrics['boundary_f1'] = self._aggregate_metric(boundary_f1, return_individual)
        
        return metrics
    
    def compute_alpha_statistics(
        self,
        alphas: Dict[str, torch.Tensor],
        slots: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor, Dict[str, float]]]:
        """
        Compute alpha value statistics.
        
        Args:
            alphas: Dictionary with 'spatial' and 'feature' alpha values
            slots: Slot representations (B, n_slots, slot_dim)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary with alpha statistics
        """
        metrics = {}
        
        for alpha_type in ['spatial', 'feature']:
            if alpha_type in alphas:
                alpha_vals = alphas[alpha_type]  # (B, n_slots, 1)
                
                # Basic statistics
                alpha_stats = {
                    'mean': alpha_vals.mean().item(),
                    'std': alpha_vals.std().item(),
                    'min': alpha_vals.min().item(),
                    'max': alpha_vals.max().item()
                }
                
                # Per-batch statistics if requested
                if return_individual:
                    alpha_stats.update({
                        'per_batch_mean': alpha_vals.mean(dim=(1, 2)),
                        'per_batch_std': alpha_vals.std(dim=(1, 2))
                    })
                
                # Alpha consistency across slots
                consistency = self._compute_alpha_consistency(alpha_vals)
                alpha_stats['consistency'] = self._aggregate_metric(
                    consistency, return_individual
                )
                
                # Alpha gradient magnitudes (for stability analysis)
                if alpha_vals.requires_grad:
                    grad_magnitude = self._compute_alpha_gradient_magnitude(alpha_vals)
                    alpha_stats['gradient_magnitude'] = grad_magnitude
                
                metrics[f'{alpha_type}_alpha'] = alpha_stats
        
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
            Dictionary with reconstruction quality metrics
        """
        metrics = {}
        
        # PSNR
        psnr_scores = self._compute_psnr(reconstruction, target)
        metrics['psnr'] = self._aggregate_metric(psnr_scores, return_individual)
        
        # SSIM (if enabled)
        if self.compute_expensive:
            ssim_scores = self._compute_ssim(reconstruction, target)
            metrics['ssim'] = self._aggregate_metric(ssim_scores, return_individual)
        else:
            metrics['ssim'] = 0.0
        
        # MSE
        mse_scores = F.mse_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))
        metrics['mse'] = self._aggregate_metric(mse_scores, return_individual)
        
        # L1 loss
        l1_scores = F.l1_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))
        metrics['l1'] = self._aggregate_metric(l1_scores, return_individual)
        
        # Perceptual loss (placeholder - would need pre-trained features)
        metrics['perceptual_loss'] = 0.0
        
        return metrics
    
    def compute_binding_interpretability(
        self,
        spatial_attention: torch.Tensor,
        masks: torch.Tensor,
        bound_features: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Compute binding interpretability metrics.
        
        Args:
            spatial_attention: Spatial attention maps (B, n_slots, H, W)
            masks: Object masks (B, n_slots, H, W)
            bound_features: Bound features (B, n_slots, slot_dim)
            return_individual: Whether to return per-sample metrics
            
        Returns:
            Dictionary with interpretability metrics
        """
        metrics = {}
        
        # Attention map clarity (entropy-based)
        attention_clarity = self._compute_attention_clarity(spatial_attention)
        metrics['attention_clarity'] = self._aggregate_metric(
            attention_clarity, return_individual
        )
        
        # Object boundary sharpness
        boundary_sharpness = self._compute_boundary_sharpness(masks)
        metrics['boundary_sharpness'] = self._aggregate_metric(
            boundary_sharpness, return_individual
        )
        
        # Feature binding coherence
        feature_coherence = self._compute_feature_coherence(
            spatial_attention, bound_features
        )
        metrics['feature_coherence'] = self._aggregate_metric(
            feature_coherence, return_individual
        )
        
        # Separation quality between objects
        object_separation = self._compute_object_separation(spatial_attention)
        metrics['object_separation'] = self._aggregate_metric(
            object_separation, return_individual
        )
        
        return metrics
    
    # Helper methods for specific metric computations
    
    def _compute_spatial_consistency(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Compute spatial consistency for each object binding."""
        B, n_slots, H, W = spatial_attention.shape
        
        # For each slot, compute how consistently it binds across spatial locations
        # Higher values when attention is focused, lower when scattered
        consistency_scores = []
        
        for b in range(B):
            batch_consistency = []
            for s in range(n_slots):
                attention_map = spatial_attention[b, s]  # (H, W)
                
                # Compute entropy of attention distribution
                attention_flat = attention_map.flatten()
                attention_norm = attention_flat / (attention_flat.sum() + self.eps)
                
                # Low entropy = high consistency (focused attention)
                entropy = -(attention_norm * torch.log(attention_norm + self.eps)).sum()
                max_entropy = np.log(H * W)
                consistency = 1.0 - (entropy / max_entropy)
                
                batch_consistency.append(consistency)
            
            consistency_scores.append(torch.stack(batch_consistency).mean())
        
        return torch.stack(consistency_scores)
    
    def _compute_cross_modal_consistency(
        self,
        spatial_attention: torch.Tensor,
        bound_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency between spatial and feature binding."""
        B, n_slots, slot_dim = bound_features.shape
        
        # Compute feature-based similarity matrix
        feature_similarities = []
        
        for b in range(B):
            features = bound_features[b]  # (n_slots, slot_dim)
            # Normalize features
            features_norm = F.normalize(features, dim=-1)
            # Compute cosine similarities
            sim_matrix = torch.mm(features_norm, features_norm.t())
            feature_similarities.append(sim_matrix)
        
        # Compute spatial overlap matrix
        spatial_overlaps = []
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            overlap_matrix = torch.zeros(n_slots, n_slots, device=attention.device)
            
            for i in range(n_slots):
                for j in range(n_slots):
                    if i != j:
                        # Compute overlap between attention maps
                        att_i = attention[i].flatten()
                        att_j = attention[j].flatten()
                        overlap = (att_i * att_j).sum() / (att_i.norm() * att_j.norm() + self.eps)
                        overlap_matrix[i, j] = overlap
            
            spatial_overlaps.append(overlap_matrix)
        
        # Compute consistency between feature similarity and spatial overlap
        consistency_scores = []
        for b in range(B):
            feat_sim = feature_similarities[b]
            spat_overlap = spatial_overlaps[b]
            
            # High feature similarity should correspond to high spatial overlap
            correlation = F.cosine_similarity(
                feat_sim.flatten(), spat_overlap.flatten(), dim=0
            )
            consistency_scores.append(correlation)
        
        return torch.stack(consistency_scores)
    
    def _compute_binding_stability(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Compute stability of binding patterns."""
        B, n_slots, H, W = spatial_attention.shape
        
        # Measure how stable the attention patterns are
        # This is a simplified version - full implementation would track across iterations
        stability_scores = []
        
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            
            # Compute variance of attention within each slot
            attention_vars = attention.var(dim=(1, 2))  # (n_slots,)
            
            # Lower variance = more stable (focused) attention
            stability = torch.exp(-attention_vars.mean())
            stability_scores.append(stability)
        
        return torch.stack(stability_scores)
    
    def _compute_slot_usage_efficiency(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Compute how efficiently slots are being used."""
        B, n_slots, H, W = spatial_attention.shape
        
        efficiency_scores = []
        
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            
            # Count how many slots have significant attention
            slot_usage = (attention.sum(dim=(1, 2)) > 0.01).float()  # (n_slots,)
            
            # Efficiency = fraction of slots actively used
            efficiency = slot_usage.mean()
            efficiency_scores.append(efficiency)
        
        return torch.stack(efficiency_scores)
    
    def _compute_slot_specialization(self, bound_features: torch.Tensor) -> torch.Tensor:
        """Compute how specialized each slot's representations are."""
        B, n_slots, slot_dim = bound_features.shape
        
        specialization_scores = []
        
        for b in range(B):
            features = bound_features[b]  # (n_slots, slot_dim)
            
            # Compute pairwise similarities between slots
            features_norm = F.normalize(features, dim=-1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())
            
            # Remove diagonal (self-similarities)
            mask = ~torch.eye(n_slots, dtype=torch.bool, device=features.device)
            off_diagonal_sims = similarity_matrix[mask]
            
            # Lower similarity = higher specialization
            specialization = 1.0 - off_diagonal_sims.mean()
            specialization_scores.append(specialization)
        
        return torch.stack(specialization_scores)
    
    def _compute_load_balancing(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Compute how evenly the binding load is distributed across slots."""
        B, n_slots, H, W = spatial_attention.shape
        
        balance_scores = []
        
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            
            # Compute total attention per slot
            slot_loads = attention.sum(dim=(1, 2))  # (n_slots,)
            
            # Normalize to get distribution
            slot_dist = slot_loads / (slot_loads.sum() + self.eps)
            
            # Compute entropy of distribution (higher = more balanced)
            entropy = -(slot_dist * torch.log(slot_dist + self.eps)).sum()
            max_entropy = np.log(n_slots)
            balance = entropy / max_entropy
            
            balance_scores.append(balance)
        
        return torch.stack(balance_scores)
    
    def _compute_slot_diversity(self, slots: torch.Tensor) -> torch.Tensor:
        """Compute diversity of slot representations."""
        B, n_slots, slot_dim = slots.shape
        
        diversity_scores = []
        
        for b in range(B):
            slot_features = slots[b]  # (n_slots, slot_dim)
            
            # Compute determinant of covariance matrix as diversity measure
            cov_matrix = torch.cov(slot_features.t())
            
            # Use log determinant for numerical stability
            sign, logdet = torch.slogdet(cov_matrix + self.eps * torch.eye(slot_dim, device=slots.device))
            diversity = logdet / slot_dim  # Normalize by dimension
            
            diversity_scores.append(diversity)
        
        return torch.stack(diversity_scores)
    
    def _compute_optimal_iou(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU with optimal assignment between predicted and target masks."""
        B = pred_masks.shape[0]
        iou_scores = []
        
        for b in range(B):
            pred = pred_masks[b]  # (n_slots, H, W)
            target = target_masks[b]  # (n_objects, H, W)
            
            n_pred = pred.shape[0]
            n_target = target.shape[0]
            
            # Compute IoU matrix
            iou_matrix = torch.zeros(n_pred, n_target, device=pred.device)
            
            for i in range(n_pred):
                for j in range(n_target):
                    intersection = (pred[i] * target[j]).sum()
                    union = (pred[i] + target[j] - pred[i] * target[j]).sum()
                    iou_matrix[i, j] = intersection / (union + self.eps)
            
            # Find optimal assignment using Hungarian algorithm or fallback
            iou_np = iou_matrix.detach().cpu().numpy()
            if SCIPY_AVAILABLE:
                row_ind, col_ind = linear_sum_assignment(-iou_np)
            else:
                row_ind, col_ind = self._linear_sum_assignment_fallback(-iou_np)
            
            # Compute mean IoU for optimal assignment
            optimal_iou = iou_matrix[row_ind, col_ind].mean()
            iou_scores.append(optimal_iou)
        
        return torch.stack(iou_scores)
    
    def _compute_ari_scores(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute Adjusted Rand Index for segmentation quality."""
        B, n_slots, H, W = pred_masks.shape
        ari_scores = []
        
        for b in range(B):
            pred = pred_masks[b]  # (n_slots, H, W)
            target = target_masks[b]  # (n_objects, H, W)
            
            # Convert to labels
            pred_labels = pred.argmax(dim=0).flatten().cpu().numpy()
            target_labels = target.argmax(dim=0).flatten().cpu().numpy()
            
            # Compute ARI
            if SKLEARN_AVAILABLE:
                ari = adjusted_rand_score(target_labels, pred_labels)
            else:
                ari = self._adjusted_rand_score_fallback(target_labels, pred_labels)
            ari_scores.append(torch.tensor(ari, device=pred.device))
        
        return torch.stack(ari_scores)
    
    def _compute_mask_coverage_completeness(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coverage and completeness of predicted masks."""
        B = pred_masks.shape[0]
        coverage_scores = []
        completeness_scores = []
        
        for b in range(B):
            pred = pred_masks[b]  # (n_slots, H, W)
            target = target_masks[b]  # (n_objects, H, W)
            
            # Coverage: fraction of predicted regions that overlap with ground truth
            pred_union = pred.sum(dim=0) > 0.5  # (H, W)
            target_union = target.sum(dim=0) > 0.5  # (H, W)
            
            coverage = (pred_union & target_union).sum().float() / (pred_union.sum().float() + self.eps)
            coverage_scores.append(coverage)
            
            # Completeness: fraction of ground truth covered by predictions
            completeness = (pred_union & target_union).sum().float() / (target_union.sum().float() + self.eps)
            completeness_scores.append(completeness)
        
        return torch.stack(coverage_scores), torch.stack(completeness_scores)
    
    def _compute_boundary_f1(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute F1 score for boundary accuracy."""
        B = pred_masks.shape[0]
        f1_scores = []
        
        # Simple edge detection using gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        for b in range(B):
            pred = pred_masks[b]  # (n_slots, H, W)
            target = target_masks[b]  # (n_objects, H, W)
            
            # Convert to binary maps and combine all objects
            pred_combined = (pred.sum(dim=0) > 0.5).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            target_combined = (target.sum(dim=0) > 0.5).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Compute edges for combined masks
            pred_edges = self._compute_edges(pred_combined, sobel_x, sobel_y)
            target_edges = self._compute_edges(target_combined, sobel_x, sobel_y)
            
            # Compute F1 between edge maps
            pred_edges_flat = pred_edges.flatten()
            target_edges_flat = target_edges.flatten()
            
            tp = (pred_edges_flat * target_edges_flat).sum()
            fp = (pred_edges_flat * (1 - target_edges_flat)).sum()
            fn = ((1 - pred_edges_flat) * target_edges_flat).sum()
            
            precision = tp / (tp + fp + self.eps)
            recall = tp / (tp + fn + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            
            f1_scores.append(f1)
        
        return torch.stack(f1_scores)
    
    def _compute_edges(
        self,
        masks: torch.Tensor,
        sobel_x: torch.Tensor,
        sobel_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge maps using Sobel filters."""
        device = masks.device
        sobel_x = sobel_x.to(device).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.to(device).unsqueeze(0).unsqueeze(0)
        
        edges_x = F.conv2d(masks, sobel_x, padding=1)
        edges_y = F.conv2d(masks, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        return (edges > 0.1).float()
    
    def _compute_alpha_consistency(self, alpha_vals: torch.Tensor) -> torch.Tensor:
        """Compute consistency of alpha values across slots and batches."""
        B, n_slots, _ = alpha_vals.shape
        
        consistency_scores = []
        
        for b in range(B):
            alphas = alpha_vals[b, :, 0]  # (n_slots,)
            
            # Consistency = 1 - coefficient of variation
            mean_alpha = alphas.mean()
            std_alpha = alphas.std()
            cv = std_alpha / (mean_alpha + self.eps)
            consistency = torch.exp(-cv)  # Convert to [0, 1] range
            
            consistency_scores.append(consistency)
        
        return torch.stack(consistency_scores)
    
    def _compute_alpha_gradient_magnitude(self, alpha_vals: torch.Tensor) -> float:
        """Compute average gradient magnitude for alpha values."""
        if not alpha_vals.requires_grad:
            return 0.0
        
        # This would typically be called during backward pass
        grad_magnitude = 0.0
        if alpha_vals.grad is not None:
            grad_magnitude = alpha_vals.grad.abs().mean().item()
        
        return grad_magnitude
    
    def _compute_psnr(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(reconstruction, target, reduction='none').mean(dim=(1, 2, 3))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + self.eps))
        return psnr
    
    def _compute_ssim(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Structural Similarity Index (simplified implementation)."""
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
            
            # Compute SSIM components
            mu1 = recon.mean()
            mu2 = tgt.mean()
            
            sigma1_sq = ((recon - mu1) ** 2).mean()
            sigma2_sq = ((tgt - mu2) ** 2).mean()
            sigma12 = ((recon - mu1) * (tgt - mu2)).mean()
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
            
            ssim_scores.append(ssim)
        
        return torch.stack(ssim_scores)
    
    def _compute_attention_clarity(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Compute clarity of attention maps (lower entropy = clearer)."""
        B, n_slots, H, W = spatial_attention.shape
        clarity_scores = []
        
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            slot_clarities = []
            
            for s in range(n_slots):
                att_map = attention[s].flatten()
                att_norm = att_map / (att_map.sum() + self.eps)
                
                # Compute entropy
                entropy = -(att_norm * torch.log(att_norm + self.eps)).sum()
                max_entropy = np.log(H * W)
                clarity = 1.0 - (entropy / max_entropy)
                
                slot_clarities.append(clarity)
            
            clarity_scores.append(torch.stack(slot_clarities).mean())
        
        return torch.stack(clarity_scores)
    
    def _compute_boundary_sharpness(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute sharpness of object boundaries."""
        B, n_slots, H, W = masks.shape
        sharpness_scores = []
        
        # Compute gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=masks.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=masks.device)
        
        for b in range(B):
            batch_masks = masks[b].unsqueeze(1)  # (n_slots, 1, H, W)
            
            # Compute edges
            edges_x = F.conv2d(batch_masks, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            edges_y = F.conv2d(batch_masks, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
            
            # Average edge magnitude as sharpness measure
            sharpness = edge_magnitude.mean()
            sharpness_scores.append(sharpness)
        
        return torch.stack(sharpness_scores)
    
    def _compute_feature_coherence(
        self,
        spatial_attention: torch.Tensor,
        bound_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute coherence between spatial binding and feature representations."""
        B, n_slots, slot_dim = bound_features.shape
        coherence_scores = []
        
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            features = bound_features[b]  # (n_slots, slot_dim)
            
            # Compute spatial compactness for each slot
            spatial_compactness = []
            for s in range(n_slots):
                att_map = attention[s]
                # Compute center of mass
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(att_map.shape[0], device=att_map.device),
                    torch.arange(att_map.shape[1], device=att_map.device),
                    indexing='ij'
                )
                
                total_weight = att_map.sum() + self.eps
                center_y = (att_map * y_coords).sum() / total_weight
                center_x = (att_map * x_coords).sum() / total_weight
                
                # Compute variance around center
                var_y = (att_map * (y_coords - center_y)**2).sum() / total_weight
                var_x = (att_map * (x_coords - center_x)**2).sum() / total_weight
                
                compactness = 1.0 / (1.0 + var_y + var_x)
                spatial_compactness.append(compactness)
            
            spatial_compactness = torch.stack(spatial_compactness)
            
            # Compute feature distinctiveness
            features_norm = F.normalize(features, dim=-1)
            feature_similarities = torch.mm(features_norm, features_norm.t())
            
            # Remove diagonal
            mask = ~torch.eye(n_slots, dtype=torch.bool, device=features.device)
            off_diagonal = feature_similarities[mask]
            feature_distinctiveness = 1.0 - off_diagonal.mean()
            
            # Coherence = correlation between spatial compactness and feature distinctiveness
            coherence = spatial_compactness.mean() * feature_distinctiveness
            coherence_scores.append(coherence)
        
        return torch.stack(coherence_scores)
    
    def _compute_object_separation(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Compute how well separated different objects are in attention space."""
        B, n_slots, H, W = spatial_attention.shape
        separation_scores = []
        
        for b in range(B):
            attention = spatial_attention[b]  # (n_slots, H, W)
            
            # Compute overlap between all pairs of attention maps
            overlaps = []
            for i in range(n_slots):
                for j in range(i + 1, n_slots):
                    att_i = attention[i].flatten()
                    att_j = attention[j].flatten()
                    
                    # Normalize attention maps
                    att_i_norm = att_i / (att_i.sum() + self.eps)
                    att_j_norm = att_j / (att_j.sum() + self.eps)
                    
                    # Compute overlap (intersection)
                    overlap = (torch.minimum(att_i_norm, att_j_norm)).sum()
                    overlaps.append(overlap)
            
            if overlaps:
                mean_overlap = torch.stack(overlaps).mean()
                separation = 1.0 - mean_overlap  # Higher separation = lower overlap
            else:
                separation = torch.tensor(1.0, device=attention.device)
            
            separation_scores.append(separation)
        
        return torch.stack(separation_scores)
    
    def _aggregate_metric(
        self,
        metric_values: torch.Tensor,
        return_individual: bool
    ) -> Union[float, torch.Tensor]:
        """Aggregate metric values based on return_individual flag."""
        if return_individual:
            return metric_values
        else:
            return metric_values.mean().item()


def evaluate_mvba_batch(
    model_output: Dict[str, torch.Tensor],
    target_images: torch.Tensor,
    target_masks: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate a batch of MVBA outputs with aggregate metrics.
    
    Args:
        model_output: Dictionary from MVBA forward pass
        target_images: Ground truth images
        target_masks: Optional ground truth masks
        device: Device for computations
        
    Returns:
        Dictionary with aggregate metric values
    """
    metrics_calc = MVBAMetrics(device=device, compute_expensive=False)
    return metrics_calc.compute_all_metrics(
        model_output, target_images, target_masks, return_individual=False
    )


def get_binding_summary(
    model_output: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Get a summary of key binding metrics.
    
    Args:
        model_output: Dictionary from MVBA forward pass
        
    Returns:
        Dictionary with key binding metrics
    """
    metrics_calc = MVBAMetrics(compute_expensive=False)
    
    # Extract key tensors
    spatial_attention = model_output['spatial_attention']
    bound_features = model_output['bound_features']
    alphas = model_output['alphas']
    slots = model_output['slots']
    
    # Compute subset of metrics
    binding_metrics = metrics_calc.compute_binding_consistency(
        spatial_attention, bound_features
    )
    slot_metrics = metrics_calc.compute_slot_utilization(
        slots, spatial_attention, bound_features
    )
    alpha_metrics = metrics_calc.compute_alpha_statistics(alphas, slots)
    
    # Combine into summary
    summary = {
        'spatial_consistency': binding_metrics['spatial_consistency'],
        'slot_usage_efficiency': slot_metrics['usage_efficiency'],
        'slot_specialization': slot_metrics['specialization'],
        'spatial_alpha_mean': alpha_metrics['spatial_alpha']['mean'],
        'feature_alpha_mean': alpha_metrics['feature_alpha']['mean']
    }
    
    return summary