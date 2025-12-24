"""
Algorithm utilities for L2R Plugin (power iteration, cost computation, etc.).
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def initialize_alpha(
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    balanced: bool = True,
) -> np.ndarray:
    """Initialize α from label distribution."""
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    N = len(labels)
    
    for g in range(K):
        group_mask = class_to_group[labels] == g
        prop = group_mask.sum().float().item() / max(N, 1)
        if balanced:
            alpha[g] = float(K * prop)  # For balanced: α[g] = K * P(g)
        else:
            alpha[g] = float(max(prop, 1e-12))  # For worst-group: α[g] = P(g)
    
    return alpha


@torch.no_grad()
def update_alpha_from_coverage(
    reject: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    balanced: bool = True,
) -> np.ndarray:
    """Update α from empirical coverage."""
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    accept = ~reject
    N = len(labels)
    
    if accept.sum() == 0:
        return np.ones(K, dtype=np.float64) * 0.5
    
    if class_weights is not None:
        cw = class_weights.to(labels.device)
        sample_w = cw[labels]
        total_weight = sample_w.sum().item()
        total_weight = max(total_weight, 1e-12)
        for g in range(K):
            in_group = class_to_group[labels] == g
            accepted_in_group = accept & in_group
            w_acc_g = sample_w[accepted_in_group].sum().item()
            cov_g = float(np.clip(w_acc_g / total_weight, 1e-12, 1.0))
            if balanced:
                alpha[g] = float(K * cov_g)
            else:
                alpha[g] = float(cov_g)
        return alpha
    
    for g in range(K):
        in_group = class_to_group[labels] == g
        accepted_in_group = accept & in_group
        empirical_cov = accepted_in_group.sum().float().item() / max(N, 1)
        if balanced:
            alpha[g] = float(K * np.clip(empirical_cov, 1e-6, 1.0))
        else:
            alpha[g] = float(np.clip(empirical_cov, 1e-6, 1.0))
    return alpha


@torch.no_grad()
def compute_cost_for_target_rejection(
    posterior: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    target_rejection: float,
    beta: Optional[np.ndarray] = None,
    balanced: bool = True,
) -> float:
    """Compute cost c to achieve a target rejection rate."""
    eps = 1e-12
    K = int(class_to_group.max().item() + 1)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    
    if balanced:
        # Balanced: α̂ = α / K
        alpha_hat_group = alpha_t / max(float(K), 1.0)
        alpha_hat_class = alpha_hat_group[class_to_group]
        mu_class = mu_t[class_to_group]
        inv_alpha_hat = 1.0 / alpha_hat_class.clamp(min=eps)
        max_rew = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
        thresh_base = ((inv_alpha_hat - mu_class).unsqueeze(0) * posterior).sum(dim=-1)
    else:
        # Worst-group: u = β / α, with β from input
        if beta is None:
            beta = np.ones(K, dtype=np.float64) / float(K)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
        u_group = beta_t / alpha_t.clamp(min=eps)
        u_class = u_group[class_to_group]
        mu_class = mu_t[class_to_group]
        max_rew = (posterior * u_class.unsqueeze(0)).max(dim=-1)[0]
        thresh_base = ((u_class - mu_class).unsqueeze(0) * posterior).sum(dim=-1)
    
    t = thresh_base - max_rew
    t_sorted = torch.sort(t)[0]
    q = max(0.0, min(1.0, 1.0 - float(target_rejection)))
    idx = int(round(q * (len(t_sorted) - 1)))
    return float(t_sorted[idx].item())


@torch.no_grad()
def power_iter_search(
    plugin,
    posterior: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    mu: np.ndarray,
    cost: float,
    num_iters: int,
    damping: float,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    target_rejection: Optional[float] = None,
    balanced: bool = True,
    beta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Power iteration to find optimal α for given μ."""
    from .metrics_utils import compute_metrics
    
    alpha = initialize_alpha(labels, class_to_group, balanced=balanced)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    K = int(class_to_group.max().item() + 1)
    
    for it in range(num_iters):
        alpha_hat = alpha.astype(np.float32)
        alpha_t = torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)
        c_it = cost
        
        if target_rejection is not None:
            c_it = compute_cost_for_target_rejection(
                posterior, class_to_group, alpha, mu, target_rejection,
                beta=beta, balanced=balanced
            )
        
        if balanced:
            plugin.set_params(alpha_t, mu_t, c_it)
        else:
            if beta is None:
                beta = np.ones(K, dtype=np.float64) / float(K)
            beta_t = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
            plugin.set_params(alpha_t, mu_t, beta_t, c_it)
        
        preds = plugin.predict(posterior)
        rej = plugin.reject(posterior)
        alpha_new = update_alpha_from_coverage(
            rej, labels, class_to_group, class_weights=class_weights, balanced=balanced
        )
        
        if damping > 0.0:
            alpha = (1.0 - damping) * alpha + damping * alpha_new
        else:
            alpha = alpha_new
        
        if verbose and (it % 10 == 0 or it == num_iters - 1):
            m = compute_metrics(preds, labels, rej, class_to_group, class_weights)
            print(f"   [PI] iter={it + 1} cov={m['coverage']:.3f} bal={m['balanced_error']:.4f}")
        
        if np.max(np.abs(alpha_new - alpha)) < 1e-4:
            break
    
    alpha_hat = alpha.astype(np.float32)
    alpha_t = torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)
    c_fin = cost
    if target_rejection is not None:
        c_fin = compute_cost_for_target_rejection(
            posterior, class_to_group, alpha, mu, target_rejection,
            beta=beta, balanced=balanced
        )
    
    if balanced:
        plugin.set_params(alpha_t, mu_t, c_fin)
    else:
        if beta is None:
            beta = np.ones(K, dtype=np.float64) / float(K)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
        plugin.set_params(alpha_t, mu_t, beta_t, c_fin)
    
    preds = plugin.predict(posterior)
    rej = plugin.reject(posterior)
    metrics = compute_metrics(preds, labels, rej, class_to_group, class_weights)
    return alpha, metrics

