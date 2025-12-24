"""
Plugin model classes for L2R algorithms.
"""

import torch
import torch.nn as nn
from typing import Optional


class BalancedLtRPlugin(nn.Module):
    """Balanced L2R Plugin (Theorem 1) with β_k = 1/K."""
    
    def __init__(self, class_to_group: torch.Tensor):
        super().__init__()
        self.class_to_group = class_to_group
        num_groups = int(class_to_group.max().item() + 1)
        self.register_buffer("alpha_group", torch.ones(num_groups))
        self.register_buffer("mu_group", torch.zeros(num_groups))
        self.register_buffer("cost", torch.tensor(0.0))
    
    def set_params(self, alpha_g: torch.Tensor, mu_g: torch.Tensor, cost: float):
        self.alpha_group = alpha_g.to(self.alpha_group.device)
        self.mu_group = mu_g.to(self.mu_group.device)
        self.cost = torch.tensor(float(cost), device=self.cost.device)
    
    def _alpha_class(self) -> torch.Tensor:
        return self.alpha_group[self.class_to_group]
    
    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]
    
    def _alpha_hat_class(self) -> torch.Tensor:
        # α̂_k = α_k · β_k ; for balanced β_k = 1/K ⇒ α̂ = α / K
        K = float(self.alpha_group.numel())
        alpha_hat_group = self.alpha_group / max(K, 1.0)
        return alpha_hat_group[self.class_to_group]
    
    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        reweighted = posterior / alpha_hat.unsqueeze(0)
        return reweighted.argmax(dim=-1)
    
    @torch.no_grad()
    def reject(self, posterior: torch.Tensor, cost: Optional[float] = None) -> torch.Tensor:
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        mu = self._mu_class()
        inv_alpha_hat = 1.0 / alpha_hat
        max_reweighted = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
        threshold = ((inv_alpha_hat - mu).unsqueeze(0) * posterior).sum(dim=-1)
        c = self.cost.item() if cost is None else float(cost)
        return max_reweighted < (threshold - c)


class GeneralizedLtRPlugin(nn.Module):
    """Generalized L2R Plugin (Theorem 12) with group weights β."""
    
    def __init__(self, class_to_group: torch.Tensor):
        super().__init__()
        self.class_to_group = class_to_group
        num_groups = int(class_to_group.max().item() + 1)
        self.register_buffer("alpha_group", torch.ones(num_groups))
        self.register_buffer("mu_group", torch.zeros(num_groups))
        self.register_buffer("beta_group", torch.ones(num_groups) / float(max(num_groups, 1)))
        self.register_buffer("cost", torch.tensor(0.0))
    
    def set_params(
        self,
        alpha_g: torch.Tensor,
        mu_g: torch.Tensor,
        beta_g: torch.Tensor,
        cost: float,
    ):
        self.alpha_group = alpha_g.to(self.alpha_group.device)
        self.mu_group = mu_g.to(self.mu_group.device)
        self.beta_group = beta_g.to(self.beta_group.device)
        self.cost = torch.tensor(float(cost), device=self.cost.device)
    
    def _u_class(self) -> torch.Tensor:
        eps = 1e-12
        u_group = self.beta_group / self.alpha_group.clamp(min=eps)
        return u_group[self.class_to_group]
    
    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]
    
    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        u = self._u_class().unsqueeze(0)
        return (posterior * u).argmax(dim=-1)
    
    @torch.no_grad()
    def reject(self, posterior: torch.Tensor, cost: Optional[float] = None) -> torch.Tensor:
        u = self._u_class().unsqueeze(0)
        mu = self._mu_class().unsqueeze(0)
        max_reweighted = (posterior * u).max(dim=-1)[0]
        threshold = ((u - mu).unsqueeze(0) * posterior).sum(dim=-1)
        c = self.cost.item() if cost is None else float(cost)
        return max_reweighted < (threshold - c)

