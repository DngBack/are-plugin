#!/usr/bin/env python3
"""
Balanced Plug-in with Gating (3 Experts) per "Learning to Reject Meets Long-Tail Learning"
============================================================================================

- Implements Algorithm 1 (power-iteration) over α and 1D λ grid for μ
- Optimizes (α, μ) on tunev, selects μ on val
- Evaluates on test; computes balanced error RC and AURC

Usage:
    python run_balanced_plugin_gating.py --dataset cifar100_lt_if100
    python run_balanced_plugin_gating.py --dataset inaturalist2018
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
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
    BalancedLtRPlugin,
    initialize_alpha,
    compute_cost_for_target_rejection,
    power_iter_search,
    plot_rc_dual,
    plot_tail_head_gap,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BalancedConfig:
    """Extended config for balanced plugin."""
    # Algorithm settings
    mu_lambda_grid: List[float] = field(
        default_factory=lambda: [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 11.0, 15.0, 20.0]
    )
    power_iter_iters: int = 20
    power_iter_damping: float = 0.5
    target_rejections: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )


def validate_sizes(logits: torch.Tensor, labels: torch.Tensor, split: str):
    """Validate and fix size mismatches."""
    if logits.shape[0] != labels.shape[0]:
        print(f"⚠️  WARNING: Size mismatch for {split}! Logits: {logits.shape[0]}, Labels: {labels.shape[0]}")
        min_size = min(logits.shape[0], labels.shape[0])
        print(f"    Truncating to {min_size} samples")
        return logits[:min_size], labels[:min_size]
    return logits, labels


def main():
    parser = argparse.ArgumentParser(description="Balanced L2R Plugin with Gating")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100_lt_if100",
        choices=["cifar100_lt_if100", "inaturalist2018", "imagenet_lt"],
        help="Dataset name",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If provided, all output will be saved to this file",
    )
    args = parser.parse_args()
    
    # Setup logging if log_file is provided
    original_stdout = sys.stdout
    log_file_handle = None
    
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, "w", encoding="utf-8")
        
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
        print(f"\n{'=' * 80}")
        print(f"LOGGING TO FILE: {log_path}")
        print(f"STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}\n")
    
    try:
        # Setup config
        config = setup_config(args.dataset)
        alg_config = BalancedConfig()
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Using dataset: {args.dataset}")
        print(f"  Classes: {config.num_classes}")
        print(f"  Experts: {config.expert_names}")
        
        # Load gating and data
        print("Loading gating network...")
        gating = load_gating_network(config, DEVICE)
        
        print("Loading S1 (tunev) and S2 (val) for selection/evaluation...")
        expert_logits_tunev = load_expert_logits(config.expert_names, "tunev", config, DEVICE)
        labels_tunev = load_labels("tunev", config, DEVICE)
        expert_logits_tunev, labels_tunev = validate_sizes(expert_logits_tunev, labels_tunev, "tunev")
        
        expert_logits_val = load_expert_logits(config.expert_names, "val", config, DEVICE)
        labels_val = load_labels("val", config, DEVICE)
        expert_logits_val, labels_val = validate_sizes(expert_logits_val, labels_val, "val")
        
        print("Computing mixture posteriors using gating network...")
        posterior_tunev = compute_mixture_posterior(expert_logits_tunev, gating, config, DEVICE)
        posterior_val = compute_mixture_posterior(expert_logits_val, gating, config, DEVICE)
        
        print("Building class-to-group mapping...")
        class_to_group = build_class_to_group(config, DEVICE)
        
        print("Loading class weights for importance weighting...")
        class_weights = load_class_weights(config, DEVICE)
        
        # Load test set
        expert_logits_test = load_expert_logits(config.expert_names, "test", config, DEVICE)
        labels_test = load_labels("test", config, DEVICE)
        expert_logits_test, labels_test = validate_sizes(expert_logits_test, labels_test, "test")
        posterior_test = compute_mixture_posterior(expert_logits_test, gating, config, DEVICE)
        
        # Baseline metrics
        mix_pred_test = posterior_test.argmax(dim=-1)
        dummy_reject = torch.zeros(len(labels_test), dtype=torch.bool, device=DEVICE)
        baseline_metrics = compute_metrics(mix_pred_test, labels_test, dummy_reject, class_to_group, class_weights)
        print(f"Baseline Gating balanced error (TEST) = {baseline_metrics['balanced_error']:.4f}")
        print(f"Baseline Gating group errors = {baseline_metrics['group_errors']}")
        
        # Create plugin
        plugin = BalancedLtRPlugin(class_to_group).to(DEVICE)
        
        # Main algorithm: optimize for each target rejection
        results_per_cost: List[Dict] = []
        for i, target_rej in enumerate(alg_config.target_rejections):
            print(f"\n=== Target {i + 1}/{len(alg_config.target_rejections)}: rejection={target_rej:.1f} ===")
            
            # Step 1: Optimize (alpha, mu) on tunev for each mu
            print("   Step 1: Optimizing (alpha, mu) on tunev for each mu...")
            candidates = []
            for lam in alg_config.mu_lambda_grid:
                mu = np.array([0.0, float(lam)], dtype=np.float64)
                alpha_found, _ = power_iter_search(
                    plugin,
                    posterior_tunev,
                    labels_tunev,
                    class_to_group,
                    mu=mu,
                    cost=0.0,
                    num_iters=alg_config.power_iter_iters,
                    damping=alg_config.power_iter_damping,
                    class_weights=class_weights,
                    verbose=False,
                    target_rejection=target_rej,
                    balanced=True,
                )
                candidates.append((alpha_found, mu))
                print(f"     mu={lam:5.1f}: alpha={alpha_found}")
            
            # Step 2: Select best mu based on val performance
            print("   Step 2: Selecting best mu based on val performance...")
            best = {
                "objective": float("inf"),
                "alpha": None,
                "mu": None,
                "val_metrics": None,
            }
            
            for alpha, mu in candidates:
                cost_val = compute_cost_for_target_rejection(
                    posterior_val, class_to_group, alpha, mu, target_rej, balanced=True
                )
                
                plugin.set_params(
                    torch.tensor(alpha.astype(np.float32), dtype=torch.float32, device=DEVICE),
                    torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                    float(cost_val),
                )
                preds_val = plugin.predict(posterior_val)
                rej_val = plugin.reject(posterior_val)
                m_val = compute_metrics(preds_val, labels_val, rej_val, class_to_group, class_weights)
                
                print(f"     mu={mu[1]:5.1f}: val_bal={m_val['balanced_error']:.4f} val_cov={m_val['coverage']:.3f}")
                
                if m_val["balanced_error"] < best["objective"]:
                    best = {
                        "objective": m_val["balanced_error"],
                        "alpha": alpha,
                        "mu": mu,
                        "val_metrics": m_val,
                    }
            
            # Step 3: Local refinement around best mu
            print("   Step 3: Local refinement around best mu...")
            best_lam = float(best["mu"][1])
            refine_step = 2.0
            for refine_iter in range(4):
                tried = []
                for lam in (best_lam - refine_step, best_lam + refine_step):
                    mu = np.array([0.0, float(lam)], dtype=np.float64)
                    alpha_found, _ = power_iter_search(
                        plugin,
                        posterior_tunev,
                        labels_tunev,
                        class_to_group,
                        mu=mu,
                        cost=0.0,
                        num_iters=alg_config.power_iter_iters,
                        damping=alg_config.power_iter_damping,
                        class_weights=class_weights,
                        verbose=False,
                        target_rejection=target_rej,
                        balanced=True,
                    )
                    
                    cost_val = compute_cost_for_target_rejection(
                        posterior_val, class_to_group, alpha_found, mu, target_rej, balanced=True
                    )
                    plugin.set_params(
                        torch.tensor(alpha_found.astype(np.float32), dtype=torch.float32, device=DEVICE),
                        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                        float(cost_val),
                    )
                    preds_val = plugin.predict(posterior_val)
                    rej_val = plugin.reject(posterior_val)
                    m_val = compute_metrics(preds_val, labels_val, rej_val, class_to_group, class_weights)
                    
                    tried.append((lam, m_val["balanced_error"], alpha_found, mu, m_val))
                
                lam_better, obj_better, alpha_better, mu_better, metr_better = min(tried, key=lambda x: x[1])
                if obj_better < best["objective"]:
                    best = {
                        "objective": obj_better,
                        "alpha": alpha_better,
                        "mu": mu_better,
                        "val_metrics": metr_better,
                    }
                    best_lam = lam_better
                    print(f"     Refine {refine_iter + 1}: mu={lam_better:.2f} val_bal={obj_better:.4f}")
                refine_step *= 0.5
            
            print(f"   Final selection: mu={best['mu'][1]:.2f} val_bal={best['val_metrics']['balanced_error']:.4f}")
            
            # Evaluate on test
            alpha_best = np.array(best["alpha"], dtype=np.float64)
            mu_best = np.array(best["mu"], dtype=np.float64)
            m_val = best["val_metrics"]
            
            cost_test = compute_cost_for_target_rejection(
                posterior_test, class_to_group, alpha_best, mu_best, target_rej, balanced=True
            )
            
            plugin.set_params(
                torch.tensor(alpha_best.astype(np.float32), dtype=torch.float32, device=DEVICE),
                torch.tensor(mu_best, dtype=torch.float32, device=DEVICE),
                float(cost_test),
            )
            preds_test = plugin.predict(posterior_test)
            rej_test = plugin.reject(posterior_test)
            m_test = compute_metrics(preds_test, labels_test, rej_test, class_to_group, class_weights)
            
            print(f"   TEST: bal={m_test['balanced_error']:.4f} cov={m_test['coverage']:.3f}")
            
            results_per_cost.append({
                "target_rejection": float(target_rej),
                "cost_val": float(compute_cost_for_target_rejection(
                    posterior_val, class_to_group, alpha_best, mu_best, target_rej, balanced=True
                )),
                "cost_test": float(cost_test),
                "alpha": alpha_best.tolist(),
                "mu": mu_best.tolist(),
                "val_metrics": {
                    "balanced_error": float(m_val["balanced_error"]),
                    "worst_group_error": float(m_val["worst_group_error"]),
                    "coverage": float(m_val["coverage"]),
                    "rejection_rate": float(1.0 - m_val["coverage"]),
                    "group_errors": [float(x) for x in m_val["group_errors"]],
                },
                "test_metrics": {
                    "balanced_error": float(m_test["balanced_error"]),
                    "worst_group_error": float(m_test["worst_group_error"]),
                    "coverage": float(m_test["coverage"]),
                    "rejection_rate": float(1.0 - m_test["coverage"]),
                    "group_errors": [float(x) for x in m_test["group_errors"]],
                },
            })
        
        # Build RC curves
        r_val = np.array([1.0 - r["val_metrics"]["coverage"] for r in results_per_cost])
        e_val = np.array([r["val_metrics"]["balanced_error"] for r in results_per_cost])
        w_val = np.array([r["val_metrics"]["worst_group_error"] for r in results_per_cost])
        gap_val = np.array([
            r["val_metrics"]["group_errors"][1] - r["val_metrics"]["group_errors"][0]
            for r in results_per_cost
        ])
        r_test = np.array([1.0 - r["test_metrics"]["coverage"] for r in results_per_cost])
        e_test = np.array([r["test_metrics"]["balanced_error"] for r in results_per_cost])
        w_test = np.array([r["test_metrics"]["worst_group_error"] for r in results_per_cost])
        gap_test = np.array([
            r["test_metrics"]["group_errors"][1] - r["test_metrics"]["group_errors"][0]
            for r in results_per_cost
        ])
        
        idx_v = np.argsort(r_val)
        r_val, e_val, w_val, gap_val = r_val[idx_v], e_val[idx_v], w_val[idx_v], gap_val[idx_v]
        idx_t = np.argsort(r_test)
        r_test, e_test, w_test, gap_test = r_test[idx_t], e_test[idx_t], w_test[idx_t], gap_test[idx_t]
        
        # Compute AURCs
        aurc_val_bal = float(np.trapz(e_val, r_val)) if r_val.size > 1 else float(e_val.mean() if e_val.size else 0.0)
        aurc_test_bal = float(np.trapz(e_test, r_test)) if r_test.size > 1 else float(e_test.mean() if e_test.size else 0.0)
        aurc_val_wst = float(np.trapz(w_val, r_val)) if r_val.size > 1 else float(w_val.mean() if w_val.size else 0.0)
        aurc_test_wst = float(np.trapz(w_test, r_test)) if r_test.size > 1 else float(w_test.mean() if w_test.size else 0.0)
        
        # Save results
        save_dict = {
            "objectives": ["balanced", "worst_group"],
            "description": "Balanced plug-in via Algorithm 1. Uses 3 experts with gating network.",
            "method": "plug-in_balanced_val_selection_gating",
            "experts": config.expert_names,
            "results_per_cost": results_per_cost,
            "rc_curve": {
                "val": {
                    "rejection_rates": r_val.tolist(),
                    "balanced_errors": e_val.tolist(),
                    "worst_group_errors": w_val.tolist(),
                    "tail_minus_head": gap_val.tolist(),
                    "aurc_balanced": aurc_val_bal,
                    "aurc_worst_group": aurc_val_wst,
                },
                "test": {
                    "rejection_rates": r_test.tolist(),
                    "balanced_errors": e_test.tolist(),
                    "worst_group_errors": w_test.tolist(),
                    "tail_minus_head": gap_test.tolist(),
                    "aurc_balanced": aurc_test_bal,
                    "aurc_worst_group": aurc_test_wst,
                },
            },
        }
        
        out_json = Path(config.results_dir) / "ltr_plugin_gating_balanced.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)
        print(f"Saved results to: {out_json}")
        
        # Export CSV
        csv_data = []
        for r in results_per_cost:
            csv_data.append({
                "target_rejection": r["target_rejection"],
                "val_balanced_error": r["val_metrics"]["balanced_error"],
                "val_worst_group_error": r["val_metrics"]["worst_group_error"],
                "val_coverage": r["val_metrics"]["coverage"],
                "val_head_error": r["val_metrics"]["group_errors"][0] if len(r["val_metrics"]["group_errors"]) >= 2 else None,
                "val_tail_error": r["val_metrics"]["group_errors"][1] if len(r["val_metrics"]["group_errors"]) >= 2 else None,
                "test_balanced_error": r["test_metrics"]["balanced_error"],
                "test_worst_group_error": r["test_metrics"]["worst_group_error"],
                "test_coverage": r["test_metrics"]["coverage"],
                "test_head_error": r["test_metrics"]["group_errors"][0] if len(r["test_metrics"]["group_errors"]) >= 2 else None,
                "test_tail_error": r["test_metrics"]["group_errors"][1] if len(r["test_metrics"]["group_errors"]) >= 2 else None,
                "alpha_head": r["alpha"][0] if len(r["alpha"]) >= 2 else None,
                "alpha_tail": r["alpha"][1] if len(r["alpha"]) >= 2 else None,
                "mu_head": r["mu"][0] if len(r["mu"]) >= 2 else None,
                "mu_tail": r["mu"][1] if len(r["mu"]) >= 2 else None,
                "cost_val": r["cost_val"],
                "cost_test": r["cost_test"],
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = Path(config.results_dir) / "ltr_plugin_gating_balanced.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV results to: {csv_path}")
        
        # Print AURCs
        print(f"Val AURC - Balanced: {aurc_val_bal:.4f} | Worst-group: {aurc_val_wst:.4f}")
        print(f"Test AURC - Balanced: {aurc_test_bal:.4f} | Worst-group: {aurc_test_wst:.4f}")
        
        # Plot
        plot_path = Path(config.results_dir) / "ltr_rc_curves_balanced_gating_test.png"
        plot_rc_dual(r_test, e_test, w_test, aurc_test_bal, aurc_test_wst, plot_path)
        print(f"Saved plot to: {plot_path}")
        
        gap_plot_path = Path(config.results_dir) / "ltr_tail_minus_head_gating_test.png"
        plot_tail_head_gap(r_test, gap_test, gap_plot_path)
        print(f"Saved gap plot to: {gap_plot_path}")
        
    finally:
        if log_file_handle is not None:
            sys.stdout = original_stdout
            log_file_handle.close()
            print(f"\n[Log saved to: {args.log_file}]")


if __name__ == "__main__":
    main()
