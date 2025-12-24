#!/usr/bin/env python3
"""
Analyze ARE Better Cases with LogitAdjust Usage
================================================

Thống kê các trường hợp mà CE only từ chối sai trong khi ARE + Gating từ chối đúng,
đặc biệt là các trường hợp mà ARE sử dụng LogitAdjust trong top-k=2.

Quét qua tất cả các rejection rates.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.gating_network_map import GatingNetwork, GatingMLP
from src.infer.loaders import (
    load_class_to_group,
    load_gating_network,
    load_plugin_params,
    EXPERT_NAMES,
    DEVICE,
    NUM_CLASSES,
    RESULTS_DIR,
)

# Configuration
DATASET = "cifar100_lt_if100"
OUTPUT_DIR = Path("./results/are_better_cases_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOGITADJUST_IDX = EXPERT_NAMES.index("logitadjust_baseline")  # Thường là 1


def load_test_logits_and_labels() -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Load test set logits and labels (using pre-computed logits)."""
    from src.infer.loaders import SPLITS_DIR
    
    print("\n1. Loading test set logits and labels...")
    
    # Load test indices first
    indices_file = SPLITS_DIR / "test_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        test_indices = json.load(f)
    
    # Load expert logits
    expert_logits_list = []
    for expert_name in EXPERT_NAMES:
        logits_path = Path(f"./outputs/logits/{DATASET}/{expert_name}/test_logits.pt")
        if not logits_path.exists():
            raise FileNotFoundError(f"Missing logits: {logits_path}")
        logits = torch.load(logits_path, map_location=DEVICE).float()
        expert_logits_list.append(logits)
    
    # Stack: [E, N, C] -> [N, E, C]
    expert_logits = torch.stack(expert_logits_list, dim=0).transpose(0, 1)
    
    # Load labels
    labels_path = Path(f"./outputs/logits/{DATASET}/{EXPERT_NAMES[0]}/test_targets.pt")
    if labels_path.exists():
        labels = torch.load(labels_path, map_location=DEVICE).long()
    else:
        # Fallback: load from dataset
        import torchvision
        dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
        labels = torch.tensor([dataset.targets[i] for i in test_indices], dtype=torch.long, device=DEVICE)
    
    print(f"   Loaded {len(expert_logits)} test samples")
    print(f"   Expert logits shape: {expert_logits.shape}")
    
    return expert_logits, labels, test_indices


def get_top_k_experts(gating_weights: np.ndarray, top_k: int = 2) -> List[int]:
    """Get top-k expert indices from gating weights."""
    top_k_indices = np.argsort(gating_weights)[-top_k:][::-1]  # Descending order
    return top_k_indices.tolist()


def compute_gating_weights_from_logits(
    expert_logits: torch.Tensor,
    gating_network: GatingNetwork
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gating weights and mixture posterior from expert logits."""
    from src.models.gating import GatingFeatureBuilder
    
    # Compute expert posteriors
    expert_posteriors = F.softmax(expert_logits, dim=-1)  # [N, E, C]
    
    # Build features and get gating weights
    feat_builder = GatingFeatureBuilder()
    features = feat_builder(expert_logits)  # [N, 7*E+3]
    gating_logits = gating_network.mlp(features)  # [N, E]
    gating_weights = gating_network.router(gating_logits)  # [N, E]
    
    # Compute mixture posterior
    mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)  # [N, C]
    
    return gating_weights, mixture_posterior


def analyze_rejection_rate(
    expert_logits: torch.Tensor,
    labels: torch.Tensor,
    test_indices: List[int],
    class_to_group: torch.Tensor,
    gating_network: GatingNetwork,
    mode: str,
    rejection_rate: float,
    top_k: int = 2,
    save_images: bool = False,
    save_dir: Path = None
) -> Dict:
    """
    Analyze cases where ARE is better than CE for a specific rejection rate.
    
    Returns:
        Dictionary with statistics
    """
    print(f"\n   Analyzing mode={mode}, rejection_rate={rejection_rate:.2f}...")
    
    # Load plugin params
    try:
        ce_alpha, ce_mu, ce_cost = load_plugin_params(
            method="ce_only", mode=mode, rejection_rate=rejection_rate
        )
        moe_alpha, moe_mu, moe_cost = load_plugin_params(
            method="moe", mode=mode, rejection_rate=rejection_rate
        )
    except Exception as e:
        print(f"   ⚠️  Error loading params: {e}")
        return None
    
    class_to_group_np = class_to_group.cpu().numpy()
    N = len(expert_logits)
    
    # Compute gating weights and mixture posterior from logits
    print(f"      Computing gating weights and mixture posterior...")
    gating_weights, mixture_posterior = compute_gating_weights_from_logits(
        expert_logits, gating_network
    )
    
    # Get CE expert logits (first expert)
    ce_logits = expert_logits[:, 0, :]  # [N, C]
    ce_posterior = F.softmax(ce_logits, dim=-1)  # [N, C]
    
    # Convert to numpy for plugin computation (detach to remove grad requirement)
    ce_posterior_np = ce_posterior.detach().cpu().numpy()
    mixture_posterior_np = mixture_posterior.detach().cpu().numpy()
    gating_weights_np = gating_weights.detach().cpu().numpy()
    
    # Import plugin classes
    from src.infer.pipeline import BalancedLtRPlugin, GeneralizedLtRPlugin
    
    # Create plugins
    if mode == "worst":
        # For worst mode, we need beta - but we'll use balanced plugin for now
        # since we don't have beta in the results
        ce_plugin = BalancedLtRPlugin(class_to_group, ce_alpha, ce_mu, ce_cost)
        are_plugin = BalancedLtRPlugin(class_to_group, moe_alpha, moe_mu, moe_cost)
    else:
        ce_plugin = BalancedLtRPlugin(class_to_group, ce_alpha, ce_mu, ce_cost)
        are_plugin = BalancedLtRPlugin(class_to_group, moe_alpha, moe_mu, moe_cost)
    
    # Statistics
    stats = {
        "rejection_rate": rejection_rate,
        "mode": mode,
        "total_samples": N,
        "cases_are_better": 0,
        "cases_with_logitadjust": 0,
        "cases_are_better_with_logitadjust": 0,
        "case_types": {
            "ce_false_reject_are_true_accept": 0,
            "ce_false_accept_are_true_reject": 0,
        },
        "case_types_with_la": {
            "ce_false_reject_are_true_accept": 0,
            "ce_false_accept_are_true_reject": 0,
        },
        "by_group": {
            "head": {
                "total": 0,
                "with_la": 0,
                "ce_false_reject_are_true_accept": 0,
                "ce_false_accept_are_true_reject": 0,
            },
            "tail": {
                "total": 0,
                "with_la": 0,
                "ce_false_reject_are_true_accept": 0,
                "ce_false_accept_are_true_reject": 0,
            },
        },
        "expert_combinations": defaultdict(int),
        "saved_image_indices": [],  # Store indices of saved images
    }
    
    # Load dataset for saving images if needed
    dataset = None
    if save_images and save_dir is not None:
        import torchvision
        dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"      Processing {N} samples...")
    for idx in tqdm(range(N), desc=f"  Processing samples", leave=False):
        true_label = labels[idx].item()
        
        # CE predictions and decisions
        ce_posterior_sample = torch.tensor(ce_posterior_np[idx], device=DEVICE)
        ce_pred = ce_posterior_sample.argmax().item()
        ce_reject = ce_plugin.reject(ce_posterior_sample.unsqueeze(0))
        ce_accept = not ce_reject
        is_correct_ce = ce_pred == true_label
        
        # ARE predictions and decisions
        mixture_posterior_sample = torch.tensor(mixture_posterior_np[idx], device=DEVICE)
        are_pred = mixture_posterior_sample.argmax().item()
        are_reject = are_plugin.reject(mixture_posterior_sample.unsqueeze(0))
        are_accept = not are_reject
        is_correct_are = are_pred == true_label
        
        # Get gating weights for this sample
        gating_weights_sample = gating_weights_np[idx]  # [E]
        top_k_experts = get_top_k_experts(gating_weights_sample, top_k)
        has_logitadjust = LOGITADJUST_IDX in top_k_experts
        
        # Check if ARE is better
        case_type = None
        
        # Case 1: CE False Reject (reject correct), ARE True Accept (accept correct)
        if (not ce_accept) and is_correct_ce and are_accept and is_correct_are:
            case_type = "ce_false_reject_are_true_accept"
        
        # Case 2: CE False Accept (accept wrong), ARE True Reject (reject wrong)
        elif ce_accept and not is_correct_ce and (not are_accept) and (not is_correct_are):
            case_type = "ce_false_accept_are_true_reject"
        
        if case_type is not None:
            stats["cases_are_better"] += 1
            
            # Check group
            group_idx = class_to_group_np[true_label]
            group_str = "tail" if group_idx == 1 else "head"
            stats["by_group"][group_str]["total"] += 1
            stats["case_types"][case_type] += 1
            stats["by_group"][group_str][case_type] += 1
            
            # Check LogitAdjust usage
            if has_logitadjust:
                stats["cases_with_logitadjust"] += 1
                stats["cases_are_better_with_logitadjust"] += 1
                stats["case_types_with_la"][case_type] += 1
                stats["by_group"][group_str]["with_la"] += 1
                
                # Record expert combination
                combo_names = [EXPERT_NAMES[i] for i in top_k_experts]
                combo_str = " + ".join(sorted(combo_names))
                stats["expert_combinations"][combo_str] += 1
                
                # Save image if requested
                if save_images and save_dir is not None and dataset is not None:
                    sample_idx = test_indices[idx]
                    class_name = dataset.classes[true_label].replace(" ", "_")
                    
                    # Create filename
                    filename = f"{group_str}_{true_label}_{class_name}_idx{sample_idx}_case{case_type}.png"
                    filepath = save_dir / filename
                    
                    # Load and save raw image
                    img, _ = dataset[sample_idx]
                    img.save(filepath)
                    stats["saved_image_indices"].append(sample_idx)
    
    # Convert defaultdict to regular dict for JSON serialization
    stats["expert_combinations"] = dict(stats["expert_combinations"])
    
    return stats


def get_all_rejection_rates(mode: str) -> List[float]:
    """Get all available rejection rates from JSON files."""
    if mode == "worst":
        results_path = RESULTS_DIR / "ltr_plugin_gating_worst.json"
        results_key = "results_per_point"
        metrics_key = "test_metrics"
    elif mode == "balanced":
        results_path = RESULTS_DIR / "ltr_plugin_gating_balanced.json"
        results_key = "results_per_cost"
        metrics_key = "val_metrics"
    else:
        return []
    
    if not results_path.exists():
        return []
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    results_list = results.get(results_key, [])
    rejection_rates = []
    
    for r in results_list:
        rejection_rate = 1.0 - r[metrics_key]["coverage"]
        rejection_rates.append(rejection_rate)
    
    return sorted(set(rejection_rates))


def main():
    """Main function."""
    print("=" * 80)
    print("ARE Better Cases Analysis with LogitAdjust Usage")
    print("=" * 80)
    
    # Load models
    print("\n0. Loading models...")
    gating_network = load_gating_network()
    class_to_group = load_class_to_group()
    
    # Load test set logits and labels (using pre-computed logits)
    expert_logits, labels, test_indices = load_test_logits_and_labels()
    
    # Get top_k from gating network
    top_k = 2
    if hasattr(gating_network, 'router') and hasattr(gating_network.router, 'top_k'):
        top_k = gating_network.router.top_k
    print(f"✓ Using top_k={top_k}")
    
    modes = ["balanced", "worst"]
    all_stats = {}
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Processing {mode.upper()} mode")
        print(f"{'='*80}")
        
        # Get all rejection rates
        rejection_rates = get_all_rejection_rates(mode)
        print(f"Found {len(rejection_rates)} rejection rates: {rejection_rates}")
        
        if len(rejection_rates) == 0:
            print(f"   ⚠️  No rejection rates found for {mode} mode")
            continue
        
        mode_stats = []
        
        for rejection_rate in rejection_rates:
            # Save images only for rejection_rate = 0.4
            save_images = (abs(rejection_rate - 0.4) < 1e-6)
            save_dir = None
            if save_images:
                save_dir = Path(f"./infer_samples_LA/{mode}")
                print(f"      Will save images to: {save_dir}")
            
            stats = analyze_rejection_rate(
                expert_logits,
                labels,
                test_indices,
                class_to_group,
                gating_network,
                mode,
                rejection_rate,
                top_k=top_k,
                save_images=save_images,
                save_dir=save_dir
            )
            
            if stats is not None:
                mode_stats.append(stats)
                
                # Print detailed results for this rejection rate
                print(f"\n   Results for rejection_rate={rejection_rate:.2f}:")
                print(f"      Total cases where ARE is better: {stats['cases_are_better']}")
                print(f"      Cases with LogitAdjust in top-{top_k}: {stats['cases_are_better_with_logitadjust']}")
                if stats['cases_are_better'] > 0:
                    pct_la = (stats['cases_are_better_with_logitadjust'] / stats['cases_are_better']) * 100
                    print(f"      Percentage with LogitAdjust: {pct_la:.2f}%")
                
                print(f"      Case types:")
                print(f"        CE False Reject + ARE True Accept: {stats['case_types']['ce_false_reject_are_true_accept']}")
                print(f"        CE False Accept + ARE True Reject: {stats['case_types']['ce_false_accept_are_true_reject']}")
                
                print(f"      Case types with LogitAdjust:")
                print(f"        CE False Reject + ARE True Accept: {stats['case_types_with_la']['ce_false_reject_are_true_accept']}")
                print(f"        CE False Accept + ARE True Reject: {stats['case_types_with_la']['ce_false_accept_are_true_reject']}")
                
                print(f"      By group:")
                for group_name in ["head", "tail"]:
                    group_stats = stats['by_group'][group_name]
                    print(f"        {group_name.upper()}:")
                    print(f"          Total: {group_stats['total']}")
                    print(f"          With LogitAdjust: {group_stats['with_la']}")
                    if group_stats['total'] > 0:
                        pct = (group_stats['with_la'] / group_stats['total']) * 100
                        print(f"          Percentage: {pct:.2f}%")
                    print(f"          CE False Reject + ARE True Accept: {group_stats['ce_false_reject_are_true_accept']}")
                    print(f"          CE False Accept + ARE True Reject: {group_stats['ce_false_accept_are_true_reject']}")
                
                if len(stats['expert_combinations']) > 0:
                    print(f"      Expert combinations:")
                    for combo, count in sorted(stats['expert_combinations'].items(), key=lambda x: x[1], reverse=True):
                        print(f"        {combo}: {count}")
                
                # Print saved images info
                if save_images and len(stats.get('saved_image_indices', [])) > 0:
                    print(f"      Saved {len(stats['saved_image_indices'])} images to {save_dir}")
        
        all_stats[mode] = mode_stats
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for mode in modes:
        if mode not in all_stats or len(all_stats[mode]) == 0:
            continue
        
        print(f"\n{mode.upper()} Mode:")
        print("-" * 80)
        print(f"{'Rejection Rate':<20} {'Total Better':<15} {'With LA':<15} {'% With LA':<15}")
        print("-" * 80)
        
        for stats in all_stats[mode]:
            rr = stats["rejection_rate"]
            total = stats["cases_are_better"]
            with_la = stats["cases_are_better_with_logitadjust"]
            pct_la = (with_la / total * 100) if total > 0 else 0.0
            print(f"{rr:<20.2f} {total:<15} {with_la:<15} {pct_la:<15.2f}%")
        
        # Overall statistics
        total_better = sum(s["cases_are_better"] for s in all_stats[mode])
        total_with_la = sum(s["cases_are_better_with_logitadjust"] for s in all_stats[mode])
        overall_pct = (total_with_la / total_better * 100) if total_better > 0 else 0.0
        
        print("-" * 80)
        print(f"{'TOTAL':<20} {total_better:<15} {total_with_la:<15} {overall_pct:<15.2f}%")
        
        # Group-wise summary
        print(f"\nGroup-wise Statistics:")
        print(f"{'Group':<15} {'Total':<15} {'With LA':<15} {'% With LA':<15}")
        print("-" * 80)
        
        head_total = sum(s["by_group"]["head"]["total"] for s in all_stats[mode])
        head_with_la = sum(s["by_group"]["head"]["with_la"] for s in all_stats[mode])
        head_pct = (head_with_la / head_total * 100) if head_total > 0 else 0.0
        
        tail_total = sum(s["by_group"]["tail"]["total"] for s in all_stats[mode])
        tail_with_la = sum(s["by_group"]["tail"]["with_la"] for s in all_stats[mode])
        tail_pct = (tail_with_la / tail_total * 100) if tail_total > 0 else 0.0
        
        print(f"{'Head':<15} {head_total:<15} {head_with_la:<15} {head_pct:<15.2f}%")
        print(f"{'Tail':<15} {tail_total:<15} {tail_with_la:<15} {tail_pct:<15.2f}%")
        
        # Expert combinations
        print(f"\nExpert Combinations (for cases with LogitAdjust):")
        all_combos = defaultdict(int)
        for stats in all_stats[mode]:
            for combo, count in stats["expert_combinations"].items():
                all_combos[combo] += count
        
        if len(all_combos) > 0:
            print(f"{'Combination':<40} {'Count':<15}")
            print("-" * 80)
            for combo, count in sorted(all_combos.items(), key=lambda x: x[1], reverse=True):
                print(f"{combo:<40} {count:<15}")
    
    # Save to JSON
    output_path = OUTPUT_DIR / "are_better_cases_with_la_analysis.json"
    
    # Convert to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    all_stats_serializable = convert_to_serializable(all_stats)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved analysis to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

