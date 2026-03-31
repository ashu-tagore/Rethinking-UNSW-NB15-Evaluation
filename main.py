"""
main.py  --  NIDS 3.0

Entry point for the 3-stage supervised contrastive learning NIDS pipeline.
Supports multiple backbone architectures for ablation studies.

Usage:
    # FT-Transformer (default):
    python main.py --train-file data/... --test-file data/... --backbone fttransformer

    # BiLSTM ablation (same pipeline, different backbone):
    python main.py --train-file data/... --test-file data/... --backbone bilstm

    # CNN ablation:
    python main.py --train-file data/... --test-file data/... --backbone cnn

    # Original hybrid:
    python main.py --train-file data/... --test-file data/... --backbone cnn_bilstm_se

    # Skip contrastive pretraining (Stage 0 only, for direct classifier baseline):
    python main.py --train-file data/... --test-file data/... --contrastive-epochs 0
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from config import ARCH, CONTRASTIVE, CLASSIFIER, FINETUNE, TRAINING, build_config
from data_loader import (
    load_from_files, load_and_preprocess,
    get_dataloaders, get_class_counts,
)
from models import build_model, BACKBONE_REGISTRY
from losses import get_loss
from trainer import run_training


def parse_args():
    p = argparse.ArgumentParser(
        description="NIDS 3.0 -- Contrastive Learning with Swappable Backbones"
    )

    # ---- Data source ----
    data = p.add_argument_group("Data source")
    data.add_argument("--train-file", type=str, default=None)
    data.add_argument("--test-file", type=str, default=None)
    data.add_argument("--data-dir", type=str, default=None)
    data.add_argument("--test-size", type=float, default=0.20)

    # ---- Backbone selection ----
    p.add_argument(
        "--backbone", type=str, default="fttransformer",
        choices=list(BACKBONE_REGISTRY.keys()),
        help="Backbone architecture (default: fttransformer)"
    )

    # ---- FT-Transformer specific args (ignored for other backbones) ----
    arch = p.add_argument_group("FT-Transformer architecture (fttransformer only)")
    arch.add_argument("--d-model", type=int, default=ARCH["d_model"])
    arch.add_argument("--n-layers", type=int, default=ARCH["n_layers"])
    arch.add_argument("--n-heads", type=int, default=ARCH["n_heads"])
    arch.add_argument("--attn-dropout", type=float, default=ARCH["attn_dropout"])
    arch.add_argument("--embed-dim", type=int, default=ARCH["embed_dim"])

    # ---- Stage 0: Contrastive pretraining ----
    con = p.add_argument_group("Stage 0: Contrastive pretraining")
    con.add_argument("--contrastive-epochs", type=int,
                     default=CONTRASTIVE["contrastive_epochs"])
    con.add_argument("--contrastive-lr", type=float,
                     default=CONTRASTIVE["contrastive_lr"])
    con.add_argument("--contrastive-patience", type=int,
                     default=CONTRASTIVE["contrastive_patience"])
    con.add_argument("--contrastive-min-epochs", type=int,
                     default=CONTRASTIVE["contrastive_min_epochs"])
    con.add_argument("--n-per-class", type=int, default=CONTRASTIVE["n_per_class"])
    con.add_argument("--temperature", type=float, default=CONTRASTIVE["temperature"])

    # ---- Stage 1: Frozen classifier ----
    clf = p.add_argument_group("Stage 1: Classifier training (backbone frozen)")
    clf.add_argument("--classifier-epochs", type=int,
                     default=CLASSIFIER["classifier_epochs"])
    clf.add_argument("--classifier-lr", type=float,
                     default=CLASSIFIER["classifier_lr"])
    clf.add_argument("--classifier-patience", type=int,
                     default=CLASSIFIER["classifier_patience"])
    clf.add_argument("--classifier-min-epochs", type=int,
                     default=CLASSIFIER["classifier_min_epochs"])

    # ---- Stage 2: Full fine-tuning ----
    ft = p.add_argument_group("Stage 2: Full fine-tuning (all params)")
    ft.add_argument("--finetune-epochs", type=int,
                    default=FINETUNE["finetune_epochs"])
    ft.add_argument("--finetune-lr", type=float, default=FINETUNE["finetune_lr"])
    ft.add_argument("--finetune-patience", type=int,
                    default=FINETUNE["finetune_patience"])
    ft.add_argument("--finetune-min-epochs", type=int,
                    default=FINETUNE["finetune_min_epochs"])

    # ---- Shared settings ----
    p.add_argument("--loss", type=str, default=TRAINING["loss"],
                   choices=["ce", "weighted_ce", "cb_focal", "ldam"])
    p.add_argument("--batch-size", type=int, default=TRAINING["batch_size"])
    p.add_argument("--weight-decay", type=float, default=TRAINING["weight_decay"])
    p.add_argument("--dropout", type=float, default=ARCH["dropout"])
    p.add_argument("--threshold-calibration", action="store_true",
                   default=TRAINING["threshold_calibration"])
    p.add_argument("--no-threshold-calibration", dest="threshold_calibration",
                   action="store_false")
    p.add_argument("--aug-mode", type=str, default="none",
                   choices=["none", "masking"],
                   help="Contrastive augmentation strategy (default: none)")
    p.add_argument("--mask-ratio", type=float, default=0.3,
                   help="Fraction of features to mask per view (default: 0.3)")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=TRAINING["seed"])
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # ---- Load data ----
    if args.train_file and args.test_file:
        X_train, X_test, y_train, y_test, le, class_names, preprocessing = \
            load_from_files(args.train_file, args.test_file)
    elif args.data_dir:
        X_train, X_test, y_train, y_test, le, class_names, preprocessing = \
            load_and_preprocess(args.data_dir, test_size=args.test_size)
    else:
        raise ValueError("Provide either --train-file + --test-file, or --data-dir")

    n_features = X_train.shape[1]
    n_classes = len(class_names)
    class_counts = get_class_counts(y_train, n_classes)

    print(f"\nFeatures: {n_features} | Classes: {n_classes}")
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ---- Dataloaders ----
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=args.batch_size
    )

    # ---- Model ----
    model = build_model(args.backbone, n_features, n_classes, args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} ({args.backbone}) | "
          f"Params: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Loss ----
    criterion = get_loss(args.loss, class_counts).to(device)
    config = build_config(args)

    print(f"Loss: {args.loss}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}\n")

    # ---- Output directory ----
    if args.run_name:
        output_dir = os.path.join(args.output_dir, args.run_name)
    else:
        output_dir = os.path.join(
            args.output_dir,
            f"contrastive_{args.backbone}"
        )

    # ---- Train ----
    start_time = time.time()
    results = run_training(
        model, train_loader, test_loader, criterion, device,
        config, output_dir, class_names=class_names
    )
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"  FINAL RESULTS ({model.__class__.__name__} + Contrastive)")
    print("=" * 60)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Macro F1:  {results['macro_f1']:.4f}")
    print(f"  Time:      {int(minutes)}m {int(seconds)}s")
    print(f"  Output:    {output_dir}/")
    print()
    print("  Per-class F1:")
    for name, f1 in results["per_class_f1"].items():
        marker = "<<" if f1 < 0.5 else ""
        print(f"    {name:<20s}: {f1:.4f} {marker}")
    print("=" * 60)

    # ---- Save summary ----
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "run_name": args.run_name or f"contrastive_{args.backbone}",
        "backbone": args.backbone,
        "model": model.__class__.__name__,
        "n_features": n_features,
        "n_classes": n_classes,
        "loss": args.loss,
        "accuracy": float(results["accuracy"]),
        "macro_f1": float(results["macro_f1"]),
        "per_class_f1": {k: float(v) for k, v in results["per_class_f1"].items()},
        "training_time_seconds": round(elapsed, 1),
        "config": config,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Save checkpoint ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "n_features": n_features,
        "n_classes": n_classes,
        "config": config,
        "thresholds": results["thresholds"],
    }, os.path.join(output_dir, "model.pt"))

    # ---- Save preprocessing ----
    with open(os.path.join(output_dir, "preprocessing.pkl"), "wb") as f:
        pickle.dump(preprocessing, f)

    print(f"\n  Saved to {output_dir}/")


if __name__ == "__main__":
    main()