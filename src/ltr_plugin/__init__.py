"""
Shared utilities for L2R Plugin scripts.
"""

from .data_utils import (
    load_expert_logits,
    load_labels,
    load_class_weights,
    build_class_to_group,
    DATASET_CONFIGS,
    setup_config,
)
from .gating_utils import load_gating_network, compute_mixture_posterior
from .metrics_utils import compute_metrics
from .plugin_models import BalancedLtRPlugin, GeneralizedLtRPlugin
from .algorithm_utils import (
    initialize_alpha,
    update_alpha_from_coverage,
    compute_cost_for_target_rejection,
    power_iter_search,
)
from .plotting_utils import plot_rc_dual, plot_rc, plot_tail_head_gap

__all__ = [
    # Data utilities
    "load_expert_logits",
    "load_labels",
    "load_class_weights",
    "build_class_to_group",
    "DATASET_CONFIGS",
    "setup_config",
    # Gating utilities
    "load_gating_network",
    "compute_mixture_posterior",
    # Metrics
    "compute_metrics",
    # Plugin models
    "BalancedLtRPlugin",
    "GeneralizedLtRPlugin",
    # Algorithm utilities
    "initialize_alpha",
    "update_alpha_from_coverage",
    "compute_cost_for_target_rejection",
    "power_iter_search",
    # Plotting
    "plot_rc_dual",
    "plot_rc",
    "plot_tail_head_gap",
]

