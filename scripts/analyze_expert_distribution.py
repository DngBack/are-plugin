#!/usr/bin/env python3
"""
Analyze Expert Distribution in Gating Network
==============================================

Thống kê phân phối các trường hợp sử dụng expert trên toàn bộ test set:
- Số lượng samples được vote bởi từng expert (CE, LogitAdjust, BalancedSoftmax)
- Số lượng samples được vote bởi các combination của experts (top-k với k=2)
- Phân phối theo head/tail classes
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.gating_network_map import GatingNetwork, GatingMLP
from src.models.gating import GatingFeatureBuilder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
DATASET = "cifar100_lt_if100"
EXPERT_NAMES = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
EXPERT_DISPLAY_NAMES = ["CE", "LogitAdjust", "BalancedSoftmax"]
SPLITS_DIR = Path(f"./data/{DATASET}_splits_fixed")
LOGITS_DIR = Path(f"./outputs/logits/{DATASET}")
GATING_CHECKPOINT = Path(f"./checkpoints/gating_map/{DATASET}/final_gating.pth")
NUM_CLASSES = 100
TAIL_THRESHOLD = 20


def load_expert_logits(split: str = "test") -> torch.Tensor:
    """Load expert logits for all experts."""
    logits_list = []
    for expert_name in EXPERT_NAMES:
        path = LOGITS_DIR / expert_name / f"{split}_logits.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing logits: {path}")
        logits = torch.load(path, map_location=DEVICE).float()
        logits_list.append(logits)
    # Stack: [E, N, C] -> [N, E, C]
    return torch.stack(logits_list, dim=0).transpose(0, 1)


def load_labels(split: str = "test") -> torch.Tensor:
    """Load labels for the split."""
    cand = LOGITS_DIR / EXPERT_NAMES[0] / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=DEVICE)
        if isinstance(t, torch.Tensor):
            return t.to(device=DEVICE, dtype=torch.long)
    
    import torchvision
    indices_file = SPLITS_DIR / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    return torch.tensor([ds.targets[i] for i in indices], dtype=torch.long, device=DEVICE)


def build_class_to_group() -> torch.Tensor:
    """Build class to group mapping (0=head, 1=tail)."""
    counts_path = SPLITS_DIR / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(NUM_CLASSES)]
    counts = np.array(class_counts)
    tail_mask = counts <= TAIL_THRESHOLD
    class_to_group = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_to_group[tail_mask] = 1
    return torch.tensor(class_to_group, dtype=torch.long, device=DEVICE)


def load_gating_network():
    """Load gating network."""
    num_experts = len(EXPERT_NAMES)
    
    checkpoint_path = GATING_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing gating checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Extract routing config
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
        print(f"  Found checkpoint config: routing={routing}, top_k={top_k}")
    
    # Create gating network
    gating = GatingNetwork(
        num_experts=num_experts,
        num_classes=NUM_CLASSES,
        routing=routing,
        top_k=top_k,
        noise_std=noise_std,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(DEVICE)
    
    # Rebuild MLP
    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(DEVICE)
    
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    print(f"✓ Loaded gating network: routing={routing}, top_k={top_k}")
    return gating


def get_top_k_experts(gating_weights: torch.Tensor, top_k: int = 2) -> List[List[int]]:
    """
    Get top-k expert indices for each sample.
    
    Args:
        gating_weights: [N, E] tensor of gating weights
        top_k: number of top experts to select
    
    Returns:
        List of lists, each containing top-k expert indices for each sample
    """
    N, E = gating_weights.shape
    top_k = min(top_k, E)
    
    # Get top-k indices for each sample
    _, topk_indices = torch.topk(gating_weights, k=top_k, dim=-1)  # [N, K]
    
    # Convert to list of lists
    top_k_experts = []
    for i in range(N):
        experts = topk_indices[i].cpu().tolist()
        top_k_experts.append(sorted(experts))  # Sort for consistency
    
    return top_k_experts


def analyze_expert_distribution(
    expert_logits: torch.Tensor,
    gating_network: GatingNetwork,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    top_k: int = 2
) -> Dict:
    """
    Analyze expert distribution in gating network.
    
    Returns:
        Dictionary with statistics about expert usage
    """
    N, E, C = expert_logits.shape
    
    # Compute gating weights
    feat_builder = GatingFeatureBuilder()
    features = feat_builder(expert_logits)  # [N, 7*E+3]
    gating_logits = gating_network.mlp(features)  # [N, E]
    gating_weights = gating_network.router(gating_logits)  # [N, E]
    
    # Get top-k experts for each sample
    top_k_experts_list = get_top_k_experts(gating_weights, top_k)
    
    # Map class labels to groups
    groups = class_to_group[labels]  # [N]
    
    # Statistics
    stats = {
        "total_samples": N,
        "top_k": top_k,
        "expert_names": EXPERT_DISPLAY_NAMES,
        "expert_indices": {name: i for i, name in enumerate(EXPERT_DISPLAY_NAMES)},
    }
    
    # 1. Count samples where each expert appears in top-k
    expert_counts = Counter()
    for experts in top_k_experts_list:
        for expert_idx in experts:
            expert_counts[expert_idx] += 1
    
    stats["expert_usage_counts"] = {
        EXPERT_DISPLAY_NAMES[i]: expert_counts[i] for i in range(E)
    }
    stats["expert_usage_percentages"] = {
        EXPERT_DISPLAY_NAMES[i]: (expert_counts[i] / N) * 100 for i in range(E)
    }
    
    # 2. Count expert combinations (for top-k=2)
    if top_k == 2:
        combination_counts = Counter()
        for experts in top_k_experts_list:
            combo = tuple(sorted(experts))
            combination_counts[combo] += 1
        
        stats["expert_combinations"] = {}
        for combo, count in combination_counts.items():
            combo_names = [EXPERT_DISPLAY_NAMES[i] for i in combo]
            combo_str = " + ".join(combo_names)
            stats["expert_combinations"][combo_str] = {
                "count": count,
                "percentage": (count / N) * 100
            }
    
    # 3. Group-wise statistics (head vs tail)
    head_mask = groups == 0
    tail_mask = groups == 1
    
    stats["group_statistics"] = {}
    
    for group_name, mask in [("head", head_mask), ("tail", tail_mask)]:
        group_samples = mask.sum().item()
        if group_samples == 0:
            continue
        
        group_top_k_experts = [top_k_experts_list[i] for i in range(N) if mask[i]]
        
        # Expert counts for this group
        group_expert_counts = Counter()
        for experts in group_top_k_experts:
            for expert_idx in experts:
                group_expert_counts[expert_idx] += 1
        
        group_stats = {
            "total_samples": group_samples,
            "expert_usage_counts": {
                EXPERT_DISPLAY_NAMES[i]: group_expert_counts[i] for i in range(E)
            },
            "expert_usage_percentages": {
                EXPERT_DISPLAY_NAMES[i]: (group_expert_counts[i] / group_samples) * 100 for i in range(E)
            }
        }
        
        # Expert combinations for this group
        if top_k == 2:
            group_combination_counts = Counter()
            for experts in group_top_k_experts:
                combo = tuple(sorted(experts))
                group_combination_counts[combo] += 1
            
            group_stats["expert_combinations"] = {}
            for combo, count in group_combination_counts.items():
                combo_names = [EXPERT_DISPLAY_NAMES[i] for i in combo]
                combo_str = " + ".join(combo_names)
                group_stats["expert_combinations"][combo_str] = {
                    "count": count,
                    "percentage": (count / group_samples) * 100
                }
        
        stats["group_statistics"][group_name] = group_stats
    
    return stats


def print_statistics(stats: Dict):
    """Print statistics in a readable format."""
    print("\n" + "=" * 80)
    print("EXPERT DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Top-K: {stats['top_k']}")
    
    print("\n" + "-" * 80)
    print("OVERALL EXPERT USAGE")
    print("-" * 80)
    print(f"{'Expert':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 80)
    for name in stats['expert_names']:
        count = stats['expert_usage_counts'][name]
        pct = stats['expert_usage_percentages'][name]
        print(f"{name:<20} {count:<15} {pct:<15.2f}%")
    
    if 'expert_combinations' in stats:
        print("\n" + "-" * 80)
        print("EXPERT COMBINATIONS (Top-2)")
        print("-" * 80)
        print(f"{'Combination':<30} {'Count':<15} {'Percentage':<15}")
        print("-" * 80)
        for combo_str, combo_stats in sorted(stats['expert_combinations'].items(), 
                                             key=lambda x: x[1]['count'], reverse=True):
            print(f"{combo_str:<30} {combo_stats['count']:<15} {combo_stats['percentage']:<15.2f}%")
    
    print("\n" + "-" * 80)
    print("GROUP-WISE STATISTICS")
    print("-" * 80)
    
    for group_name in ["head", "tail"]:
        if group_name not in stats['group_statistics']:
            continue
        
        group_stats = stats['group_statistics'][group_name]
        print(f"\n{group_name.upper()} Classes:")
        print(f"  Total samples: {group_stats['total_samples']}")
        
        print(f"\n  Expert Usage:")
        print(f"    {'Expert':<20} {'Count':<15} {'Percentage':<15}")
        print(f"    {'-' * 50}")
        for name in stats['expert_names']:
            count = group_stats['expert_usage_counts'][name]
            pct = group_stats['expert_usage_percentages'][name]
            print(f"    {name:<20} {count:<15} {pct:<15.2f}%")
        
        if 'expert_combinations' in group_stats:
            print(f"\n  Expert Combinations:")
            print(f"    {'Combination':<30} {'Count':<15} {'Percentage':<15}")
            print(f"    {'-' * 60}")
            for combo_str, combo_stats in sorted(group_stats['expert_combinations'].items(),
                                                 key=lambda x: x[1]['count'], reverse=True):
                print(f"    {combo_str:<30} {combo_stats['count']:<15} {combo_stats['percentage']:<15.2f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    print("Loading data and models...")
    
    # Load data
    expert_logits = load_expert_logits("test")
    labels = load_labels("test")
    class_to_group = build_class_to_group()
    
    print(f"✓ Loaded {len(expert_logits)} test samples")
    print(f"✓ Expert logits shape: {expert_logits.shape}")
    
    # Load gating network
    gating_network = load_gating_network()
    
    # Get top_k from gating network
    top_k = 2
    if hasattr(gating_network, 'router') and hasattr(gating_network.router, 'top_k'):
        top_k = gating_network.router.top_k
    elif hasattr(gating_network, 'routing_type') and gating_network.routing_type == 'top_k':
        top_k = 2  # Default for top_k routing
    
    print(f"✓ Using top_k={top_k}")
    
    # Analyze
    print("\nAnalyzing expert distribution...")
    stats = analyze_expert_distribution(
        expert_logits,
        gating_network,
        labels,
        class_to_group,
        top_k=top_k
    )
    
    # Print statistics
    print_statistics(stats)
    
    # Save to JSON
    output_path = Path(f"./results/expert_distribution_{DATASET}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    stats_serializable = convert_to_json_serializable(stats)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved statistics to: {output_path}")


if __name__ == "__main__":
    main()

