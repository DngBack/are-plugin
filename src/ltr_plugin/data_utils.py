"""
Data loading utilities for L2R Plugin scripts.
"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision


# Dataset configurations
DATASET_CONFIGS = {
    "cifar100_lt_if100": {
        "splits_dir": "./data/cifar100_lt_if100_splits_fixed",
        "logits_dir": "./outputs/logits/cifar100_lt_if100",
        "gating_checkpoint": "./checkpoints/gating_map/cifar100_lt_if100/best_gating.pth",
        "results_dir": "./results/ltr_plugin/cifar100_lt_if100",
        "expert_names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
        "num_classes": 100,
        "num_groups": 2,
        "tail_threshold": 20,
    },
    "inaturalist2018": {
        "splits_dir": "./data/inaturalist2018_splits",
        "logits_dir": "./outputs/logits/inaturalist2018",
        "gating_checkpoint": "./checkpoints/gating_map/inaturalist2018/best_gating.pth",
        "results_dir": "./results/ltr_plugin/inaturalist2018",
        "expert_names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
        "num_classes": 8142,
        "num_groups": 2,
        "tail_threshold": 20,
    },
    "imagenet_lt": {
        "splits_dir": "./data/imagenet_lt_splits",
        "logits_dir": "./outputs/logits/imagenet_lt",
        "gating_checkpoint": "./checkpoints/gating_map/imagenet_lt/best_gating.pth",
        "results_dir": "./results/ltr_plugin/imagenet_lt",
        "expert_names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
        "num_classes": 1000,
        "num_groups": 2,
        "tail_threshold": 20,
    },
}


@dataclass
class Config:
    """Configuration for L2R Plugin scripts."""
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    gating_checkpoint: str = "./checkpoints/gating_map/cifar100_lt_if100/best_gating.pth"
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"
    expert_names: List[str] = field(
        default_factory=lambda: ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
    )
    num_classes: int = 100
    num_groups: int = 2
    tail_threshold: int = 20
    seed: int = 42


def setup_config(dataset_name: str) -> Config:
    """Setup Config based on dataset selection."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}"
        )
    
    ds_config = DATASET_CONFIGS[dataset_name]
    
    return Config(
        dataset_name=dataset_name,
        splits_dir=ds_config["splits_dir"],
        logits_dir=ds_config["logits_dir"],
        gating_checkpoint=ds_config["gating_checkpoint"],
        results_dir=ds_config["results_dir"],
        expert_names=ds_config["expert_names"],
        num_classes=ds_config["num_classes"],
        num_groups=ds_config["num_groups"],
        tail_threshold=ds_config.get("tail_threshold", 20),
    )


def load_expert_logits(
    expert_names: List[str],
    split: str,
    config: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Load logits from all experts and stack them."""
    logits_list = []
    
    for expert_name in expert_names:
        path = Path(config.logits_dir) / expert_name / f"{split}_logits.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing logits: {path}")
        logits = torch.load(path, map_location=device).float()
        logits_list.append(logits)
    
    # Stack: [E, N, C] -> transpose to [N, E, C]
    return torch.stack(logits_list, dim=0).transpose(0, 1)


def load_labels(
    split: str,
    config: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Load labels for a given split."""
    # Prefer saved targets alongside logits
    cand = Path(config.logits_dir) / config.expert_names[0] / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=device)
        if isinstance(t, torch.Tensor):
            return t.to(device=device, dtype=torch.long)
    
    # Fallback: load from JSON targets file (for iNaturalist/ImageNet-LT)
    targets_file = Path(config.splits_dir) / f"{split}_targets.json"
    if targets_file.exists():
        with open(targets_file, "r", encoding="utf-8") as f:
            targets = json.load(f)
        return torch.tensor(targets, dtype=torch.long, device=device)
    
    # Fallback: reconstruct from CIFAR100 and indices
    indices_file = Path(config.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    return torch.tensor(
        [ds.targets[i] for i in indices], dtype=torch.long, device=device
    )


def load_class_weights(
    config: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Load inverse class weights for importance weighting."""
    counts_path = Path(config.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(config.num_classes)]
    
    counts = np.array(class_counts, dtype=np.float64)
    total = counts.sum()
    train_probs = counts / max(total, 1e-12)
    # Test is balanced → weights = train_probs / (1/C) = train_probs * C
    weights = train_probs * config.num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_class_to_group(
    config: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Build class-to-group mapping (0=head, 1=tail)."""
    counts_path = Path(config.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(config.num_classes)]
    counts = np.array(class_counts)
    tail_mask = counts <= config.tail_threshold
    class_to_group = np.zeros(config.num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1
    return torch.tensor(class_to_group, dtype=torch.long, device=device)

