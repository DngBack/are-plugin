"""
Gating network utilities for L2R Plugin scripts.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_utils import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_gating_network(
    config: "Config",
    device: str = DEVICE,
):
    """Load trained gating network."""
    from src.models.gating_network_map import GatingNetwork, GatingMLP
    
    num_experts = len(config.expert_names)
    num_classes = config.num_classes
    
    checkpoint_path = Path(config.gating_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing gating checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract routing config from checkpoint
    routing = "dense"
    top_k = 2
    noise_std = 1.0
    hidden_dims = [256, 128]
    dropout = 0.1
    activation = "relu"
    
    if "config" in checkpoint:
        gating_config = checkpoint["config"].get("gating", {})
        routing = gating_config.get("routing", "dense")
        top_k = gating_config.get("top_k", 2)
        noise_std = gating_config.get("noise_std", 1.0)
        hidden_dims = gating_config.get("hidden_dims", [256, 128])
        dropout = gating_config.get("dropout", 0.1)
        activation = gating_config.get("activation", "relu")
    
    # Create gating network
    gating = GatingNetwork(
        num_experts=num_experts,
        num_classes=num_classes,
        routing=routing,
        top_k=top_k,
        noise_std=noise_std,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(device)
    
    # Rebuild MLP to match compact feature dimension
    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(device)
    
    # Verify checkpoint matches expected number of experts
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "mlp.mlp.8.weight" in state_dict:
            mlp_output_dim = state_dict["mlp.mlp.8.weight"].shape[0]
            if mlp_output_dim != num_experts:
                raise ValueError(
                    f"Checkpoint expects {mlp_output_dim} experts but config has {num_experts} experts. "
                    f"Please check expert_names: {config.expert_names}"
                )
    
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    return gating


@torch.no_grad()
def compute_mixture_posterior(
    expert_logits: torch.Tensor,
    gating_net,
    config: "Config",
    device: str = DEVICE,
) -> torch.Tensor:
    """Compute mixture posterior using gating network."""
    # expert_logits: [N, E, C]
    expert_posteriors = F.softmax(expert_logits, dim=-1)
    num_experts_logits = expert_logits.shape[1]
    num_experts_config = len(config.expert_names)
    if num_experts_logits != num_experts_config:
        raise ValueError(
            f"Mismatch: logits have {num_experts_logits} experts but config expects {num_experts_config} experts"
        )
    
    # Build compact features and get weights via MLP + router
    from src.models.gating import GatingFeatureBuilder
    
    feat_builder = GatingFeatureBuilder()
    features = feat_builder(expert_logits)  # [N, 7*E+3]
    gating_logits = gating_net.mlp(features)
    gating_weights = gating_net.router(gating_logits)
    
    if torch.isnan(gating_weights).any():
        N, E = expert_logits.shape[0], expert_logits.shape[1]
        gating_weights = torch.ones(N, E, device=device) / E
    
    mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)
    return mixture_posterior

