"""
Metrics computation utilities for L2R Plugin scripts.
"""

import numpy as np
import torch
from typing import Dict, Optional


@torch.no_grad()
def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    reject: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute selective classification metrics."""
    accept = ~reject
    if accept.sum() == 0:
        num_groups = int(class_to_group.max().item() + 1)
        return {
            "selective_error": 1.0,
            "coverage": 0.0,
            "group_errors": [1.0] * num_groups,
            "balanced_error": 1.0,
            "worst_group_error": 1.0,
        }
    
    preds_a = preds[accept]
    labels_a = labels[accept]
    errors = (preds_a != labels_a).float()
    selective_error = errors.mean().item()
    coverage = accept.float().mean().item()
    groups = class_to_group[labels_a]
    num_groups = int(class_to_group.max().item() + 1)
    
    group_errors = []
    
    if class_weights is not None:
        device = labels.device
        class_weights = class_weights.to(device)
        
        for g in range(num_groups):
            mask = groups == g
            if mask.sum() == 0:
                group_errors.append(1.0)
            else:
                y_g = labels_a[mask]
                preds_g = preds_a[mask]
                sample_weights = class_weights[y_g]
                errors_in_group = (preds_g != y_g).float()
                weighted_errors = (sample_weights * errors_in_group).sum().item()
                total_weight = sample_weights.sum().item()
                if total_weight > 0:
                    group_errors.append(weighted_errors / total_weight)
                else:
                    group_errors.append(1.0)
    else:
        for g in range(num_groups):
            mask = groups == g
            if mask.sum() == 0:
                group_errors.append(1.0)
            else:
                num_errors_in_group = errors[mask].sum().item()
                num_accepted_in_group = mask.sum().item()
                conditional_error = num_errors_in_group / num_accepted_in_group
                group_errors.append(conditional_error)
    
    balanced_error = float(np.mean(group_errors))
    worst_group_error = float(np.max(group_errors))
    
    return {
        "selective_error": selective_error,
        "coverage": coverage,
        "group_errors": group_errors,
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
    }

