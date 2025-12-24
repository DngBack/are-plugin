#!/usr/bin/env python3
"""
Worst-group Plug-in with Gating (3 Experts) per "Learning to Reject Meets Long-Tail Learning"
==============================================================================================

- Implements Algorithm 2 (Worst-group Plug-in) with inner Algorithm 1 (CS plug-in)
- Uses mixture posteriors from a trained gating network combining experts
- Uses tunev (S1) for optimization and val (S2) for exponentiated-gradient updates
- Evaluates on test; reports RC curves and AURC for worst-group (primary) and balanced (secondary)

Usage:
    python run_worst_plugin_gating.py --dataset cifar100_lt_if100
    python run_worst_plugin_gating.py --dataset inaturalist2018
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import pandas as pd

from src.ltr_plugin import (
    setup_config,
    load_expert_logits,
    load_labels,
    load_class_weights,
    build_class_to_group,
    load_gating_network,
    compute_mixture_posterior,
    compute_metrics,
    GeneralizedLtRPlugin,
    initialize_alpha,
    update_alpha_from_coverage,
    compute_cost_for_target_rejection,
    power_iter_search,
    plot_rc,
    plot_tail_head_gap,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class WorstConfig:
    """Extended config for worst-group plugin."""
    # Inner CS plug-in (Algorithm 1) grid over single λ = μ_tail − μ_head
    mu_lambda_grid: List[float] = field(default_factory=lambda: [1.0, 6.0, 11.0])
    power_iter_iters: int = 10
    power_iter_damping: float = 0.5
    # Algorithm 2 (Worst): EG iterations and step-size
    eg_iters: int = 25
    eg_step: float = 1.0
    # Target rejection grid
    target_rejections: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


def validate_sizes(logits: torch.Tensor, labels: torch.Tensor, split: str):
    """Validate and fix size mismatches."""
    if logits.shape[0] != labels.shape[0]:
        print(f"⚠️  WARNING: Size mismatch for {split}! Logits: {logits.shape[0]}, Labels: {labels.shape[0]}")
        min_size = min(logits.shape[0], labels.shape[0])
        print(f"    Truncating to {min_size} samples")
        return logits[:min_size], labels[:min_size]
    return logits, labels


@torch.no_grad()
def cs_plugin_inner(
    plugin: GeneralizedLtRPlugin,
    posterior_s1: torch.Tensor,
    labels_s1: torch.Tensor,
    posterior_s2: torch.Tensor,
    labels_s2: torch.Tensor,
    class_to_group: torch.Tensor,
    beta: np.ndarray,
    target_rej: float,
    class_weights,
    alg_config: WorstConfig,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Inner CS plug-in (Algorithm 1) for worst-group optimization."""
    K = int(class_to_group.max().item() + 1)
    best = {
        "obj": float("inf"),
        "alpha": None,
        "mu": None,
        "cost": None,
        "metrics": None,
    }
    
    for lam in alg_config.mu_lambda_grid:
        mu = np.array([0.0, float(lam)], dtype=np.float64) if K == 2 else np.zeros(K, dtype=np.float64)
        alpha = initialize_alpha(labels_s1, class_to_group, balanced=False)
        
        for _ in range(alg_config.power_iter_iters):
            beta_t = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
            plugin.set_params(
                torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                beta_t,
                0.0,
            )
            c_it = compute_cost_for_target_rejection(
                posterior_s1, class_to_group, alpha, mu, target_rej,
                beta=beta, balanced=False
            )
            plugin.set_params(
                torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                beta_t,
                float(c_it),
            )
            rej = plugin.reject(posterior_s1)
            alpha_new = update_alpha_from_coverage(
                rej, labels_s1, class_to_group, class_weights, balanced=False
            )
            alpha = (1.0 - alg_config.power_iter_damping) * alpha + alg_config.power_iter_damping * alpha_new
        
        c_s2 = compute_cost_for_target_rejection(
            posterior_s2, class_to_group, alpha, mu, target_rej,
            beta=beta, balanced=False
        )
        plugin.set_params(
            torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
            torch.tensor(mu, dtype=torch.float32, device=DEVICE),
            torch.tensor(beta, dtype=torch.float32, device=DEVICE),
            float(c_s2),
        )
        preds_s2 = plugin.predict(posterior_s2)
        rej_s2 = plugin.reject(posterior_s2)
        m_s2 = compute_metrics(preds_s2, labels_s2, rej_s2, class_to_group, class_weights)
        obj = float(np.sum(np.array(m_s2["group_errors"]) * np.array(beta)))
        
        if obj < best["obj"]:
            best = {
                "obj": obj,
                "alpha": alpha.copy(),
                "mu": mu.copy(),
                "cost": float(c_s2),
                "metrics": m_s2,
            }
    
    # Local refinement around best μ (λ)
    if K == 2 and best["mu"] is not None:
        base_lam = float(best["mu"][1])
        step = 2.0
        for _ in range(4):
            tried = []
            for lam in (base_lam - step, base_lam + step):
                mu = np.array([0.0, float(lam)], dtype=np.float64)
                alpha = initialize_alpha(labels_s1, class_to_group, balanced=False)
                
                for _ in range(max(8, alg_config.power_iter_iters // 2)):
                    plugin.set_params(
                        torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                        torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                        0.0,
                    )
                    c_it = compute_cost_for_target_rejection(
                        posterior_s1, class_to_group, alpha, mu, target_rej,
                        beta=beta, balanced=False
                    )
                    plugin.set_params(
                        torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                        torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                        float(c_it),
                    )
                    rej = plugin.reject(posterior_s1)
                    alpha_new = update_alpha_from_coverage(
                        rej, labels_s1, class_to_group, class_weights, balanced=False
                    )
                    alpha = (1.0 - alg_config.power_iter_damping) * alpha + alg_config.power_iter_damping * alpha_new
                
                c_s2 = compute_cost_for_target_rejection(
                    posterior_s2, class_to_group, alpha, mu, target_rej,
                    beta=beta, balanced=False
                )
                plugin.set_params(
                    torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
                    torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                    torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                    float(c_s2),
                )
                preds_s2 = plugin.predict(posterior_s2)
                rej_s2 = plugin.reject(posterior_s2)
                m_s2 = compute_metrics(preds_s2, labels_s2, rej_s2, class_to_group, class_weights)
                obj = float(np.sum(np.array(m_s2["group_errors"]) * np.array(beta)))
                tried.append((lam, obj, alpha.copy(), mu.copy(), float(c_s2), m_s2))
            
            lam_b, obj_b, alpha_b, mu_b, c_b, m_b = min(tried, key=lambda x: x[1])
            if obj_b < best["obj"]:
                best = {
                    "obj": obj_b,
                    "alpha": alpha_b,
                    "mu": mu_b,
                    "cost": c_b,
                    "metrics": m_b,
                }
                base_lam = lam_b
            step *= 0.5
    
    return best["alpha"], best["mu"], best["cost"], best["metrics"]


def main():
    parser = argparse.ArgumentParser(description="Worst-group L2R Plugin with Gating")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100_lt_if100",
        choices=["cifar100_lt_if100", "inaturalist2018", "imagenet_lt"],
        help="Dataset name"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If provided, all output will be saved to this file"
    )
    args = parser.parse_args()
    
    # Setup logging if log_file is provided
    original_stdout = sys.stdout
    log_file_handle = None
    
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, 'w', encoding='utf-8')
        
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = TeeOutput(original_stdout, log_file_handle)
        print(f"\n{'='*80}")
        print(f"LOGGING TO FILE: {log_path}")
        print(f"STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    try:
        # Setup config
        config = setup_config(args.dataset)
        alg_config = WorstConfig()
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Using dataset: {args.dataset}")
        print(f"  Classes: {config.num_classes}")
        print(f"  Experts: {config.expert_names}")
        
        # Load gating and data
        gating = load_gating_network(config, DEVICE)
        
        expert_logits_s1 = load_expert_logits(config.expert_names, "tunev", config, DEVICE)
        labels_s1 = load_labels("tunev", config, DEVICE)
        expert_logits_s1, labels_s1 = validate_sizes(expert_logits_s1, labels_s1, "tunev")
        
        expert_logits_s2 = load_expert_logits(config.expert_names, "val", config, DEVICE)
        labels_s2 = load_labels("val", config, DEVICE)
        expert_logits_s2, labels_s2 = validate_sizes(expert_logits_s2, labels_s2, "val")
        
        expert_logits_test = load_expert_logits(config.expert_names, "test", config, DEVICE)
        labels_test = load_labels("test", config, DEVICE)
        expert_logits_test, labels_test = validate_sizes(expert_logits_test, labels_test, "test")
        
        post_s1 = compute_mixture_posterior(expert_logits_s1, gating, config, DEVICE)
        post_s2 = compute_mixture_posterior(expert_logits_s2, gating, config, DEVICE)
        post_test = compute_mixture_posterior(expert_logits_test, gating, config, DEVICE)
        
        class_to_group = build_class_to_group(config, DEVICE)
        class_weights = load_class_weights(config, DEVICE)
        
        plugin = GeneralizedLtRPlugin(class_to_group).to(DEVICE)
        
        # Main algorithm: EG over β for each target rejection
        results = []
        for target_rej in alg_config.target_rejections:
            # EG over β
            K = int(class_to_group.max().item() + 1)
            beta = np.ones(K, dtype=np.float64) / float(K)
            alpha_best = None
            mu_best = None
            cost_best = None
            
            for _ in range(alg_config.eg_iters):
                alpha_t, mu_t, cost_t, metr_s2 = cs_plugin_inner(
                    plugin,
                    post_s1,
                    labels_s1,
                    post_s2,
                    labels_s2,
                    class_to_group,
                    beta,
                    float(target_rej),
                    class_weights,
                    alg_config,
                )
                # EG update using group errors on S2
                e = np.array(metr_s2["group_errors"], dtype=np.float64)
                beta = beta * np.exp(alg_config.eg_step * e)
                s = max(np.sum(beta), 1e-12)
                beta = beta / s
                alpha_best, mu_best, cost_best = alpha_t, mu_t, cost_t
            
            # Evaluate on test using final params
            c_test = compute_cost_for_target_rejection(
                post_test, class_to_group, alpha_best, mu_best, float(target_rej),
                beta=beta, balanced=False
            )
            plugin.set_params(
                torch.tensor(alpha_best, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu_best, dtype=torch.float32, device=DEVICE),
                torch.tensor(beta, dtype=torch.float32, device=DEVICE),
                float(c_test),
            )
            preds_test = plugin.predict(post_test)
            rej_test = plugin.reject(post_test)
            m_test = compute_metrics(preds_test, labels_test, rej_test, class_to_group, class_weights)
            
            results.append({
                "target_rejection": float(target_rej),
                "beta": beta.tolist(),
                "alpha": alpha_best.tolist(),
                "mu": mu_best.tolist(),
                "cost_test": float(c_test),
                "test_metrics": {
                    "coverage": float(m_test["coverage"]),
                    "balanced_error": float(m_test["balanced_error"]),
                    "worst_group_error": float(m_test["worst_group_error"]),
                    "group_errors": [float(x) for x in m_test["group_errors"]],
                },
            })
        
        # Build RC curves
        r = np.array([1.0 - r_["test_metrics"]["coverage"] for r_ in results])
        ew = np.array([r_["test_metrics"]["worst_group_error"] for r_ in results])
        eb = np.array([r_["test_metrics"]["balanced_error"] for r_ in results])
        idx = np.argsort(r)
        r, ew, eb = r[idx], ew[idx], eb[idx]
        
        # Tail - Head error gap
        gap = []
        for r_ in results:
            ge = r_["test_metrics"]["group_errors"]
            if isinstance(ge, list) and len(ge) >= 2:
                gap.append(float(ge[1]) - float(ge[0]))
            else:
                gap.append(float("nan"))
        gap = np.array(gap)[idx]
        
        # Filter to only keep points with rejection rate <= 0.9
        mask = r <= 0.9
        r = r[mask]
        ew = ew[mask]
        eb = eb[mask]
        gap = gap[mask]
        
        aurc_w = float(np.trapz(ew, r)) if r.size > 1 else float(ew.mean() if ew.size else 0.0)
        aurc_b = float(np.trapz(eb, r)) if r.size > 1 else float(eb.mean() if eb.size else 0.0)
        
        # Save results
        out_json = Path(config.results_dir) / "ltr_plugin_gating_worst.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "description": "Worst-group plug-in via Algorithm 2 with inner CS plug-in (Algorithm 1). Mixture posterior from gating over 3 experts.",
                "experts": config.expert_names,
                "target_rejections": list(alg_config.target_rejections),
                "results_per_point": results,
                "rc_curve": {
                    "rejection_rates": r.tolist(),
                    "worst_group_errors": ew.tolist(),
                    "balanced_errors": eb.tolist(),
                    "tail_minus_head": gap.tolist(),
                    "aurc_worst_group": aurc_w,
                    "aurc_balanced": aurc_b,
                },
            }, f, indent=2)
        
        print(f"Saved results to: {out_json}")
        
        # Plot
        plot_path = Path(config.results_dir) / "ltr_rc_curves_balanced_worst_gating_test.png"
        plot_rc(r, ew, eb, aurc_w, aurc_b, plot_path)
        print(f"Saved plot to: {plot_path}")
        
        gap_plot_path = Path(config.results_dir) / "ltr_tail_minus_head_worst_gating_test.png"
        plot_tail_head_gap(r, gap, gap_plot_path, title="Tail-Head Error Gap vs Rejection Rate (Worst Gating)")
        print(f"Saved gap plot to: {gap_plot_path}")
        
        # Export CSV
        csv_data = []
        for r_ in results:
            csv_data.append({
                'target_rejection': r_['target_rejection'],
                'test_balanced_error': r_['test_metrics']['balanced_error'],
                'test_worst_group_error': r_['test_metrics']['worst_group_error'],
                'test_coverage': r_['test_metrics']['coverage'],
                'test_head_error': r_['test_metrics']['group_errors'][0] if len(r_['test_metrics']['group_errors']) >= 2 else None,
                'test_tail_error': r_['test_metrics']['group_errors'][1] if len(r_['test_metrics']['group_errors']) >= 2 else None,
                'test_tail_minus_head': r_['test_metrics']['group_errors'][1] - r_['test_metrics']['group_errors'][0] if len(r_['test_metrics']['group_errors']) >= 2 else None,
                'beta_head': r_['beta'][0] if len(r_['beta']) >= 2 else None,
                'beta_tail': r_['beta'][1] if len(r_['beta']) >= 2 else None,
                'alpha_head': r_['alpha'][0] if len(r_['alpha']) >= 2 else None,
                'alpha_tail': r_['alpha'][1] if len(r_['alpha']) >= 2 else None,
                'mu_head': r_['mu'][0] if len(r_['mu']) >= 2 else None,
                'mu_tail': r_['mu'][1] if len(r_['mu']) >= 2 else None,
                'cost_test': r_['cost_test'],
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = Path(config.results_dir) / "ltr_plugin_gating_worst.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV results to: {csv_path}")
        
    finally:
        if log_file_handle is not None:
            sys.stdout = original_stdout
            log_file_handle.close()
            print(f"\n[Log saved to: {args.log_file}]")


if __name__ == "__main__":
    main()
