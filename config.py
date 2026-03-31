"""
config.py  --  NIDS 3.0

Centralized hyperparameter defaults for the 3-stage contrastive pipeline.
main.py reads these as defaults; override any value via CLI flags.
"""

# ---- FT-Transformer architecture (fttransformer backbone only) -------------
ARCH = {
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "attn_dropout": 0.1,
    "embed_dim": 128,       # projection head output dim (all backbones)
    "dropout": 0.3,         # classifier head dropout (all backbones)
}

# ---- Stage 0: Contrastive pretraining --------------------------------------
CONTRASTIVE = {
    "contrastive_epochs": 100,
    "contrastive_lr": 5e-4,
    "contrastive_patience": 15,
    "contrastive_min_epochs": 20,
    "n_per_class": 8,
    "temperature": 0.07,
}

# ---- Stage 1: Frozen classifier training -----------------------------------
CLASSIFIER = {
    "classifier_epochs": 60,
    "classifier_lr": 1e-3,
    "classifier_patience": 15,
    "classifier_min_epochs": 15,
}

# ---- Stage 2: Full fine-tuning (all params unfrozen) -----------------------
FINETUNE = {
    "finetune_epochs": 60,
    "finetune_lr": 5e-4,
    "finetune_patience": 15,
    "finetune_min_epochs": 15,
}

# ---- Shared training settings ----------------------------------------------
TRAINING = {
    "batch_size": 256,
    "weight_decay": 1e-4,
    "loss": "cb_focal",
    "threshold_calibration": True,
    "seed": 42,
}


def build_config(args):
    """Merge CLI args into a flat config dict for trainer.py."""
    return {
        "contrastive": True,
        "backbone": args.backbone,

        # FT-Transformer specific (stored for inference reconstruction)
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "attn_dropout": args.attn_dropout,
        "embed_dim": args.embed_dim,

        # Stage 0
        "contrastive_epochs": args.contrastive_epochs,
        "contrastive_lr": args.contrastive_lr,
        "contrastive_patience": args.contrastive_patience,
        "contrastive_min_epochs": args.contrastive_min_epochs,
        "n_per_class": args.n_per_class,
        "temperature": args.temperature,

        # Stage 1
        "classifier_epochs": args.classifier_epochs,
        "classifier_lr": args.classifier_lr,
        "classifier_patience": args.classifier_patience,
        "classifier_min_epochs": args.classifier_min_epochs,

        # Stage 2
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "finetune_patience": args.finetune_patience,
        "finetune_min_epochs": args.finetune_min_epochs,

        # Shared
        "weight_decay": args.weight_decay,
        "threshold_calibration": args.threshold_calibration,
        "aug_mode": args.aug_mode,
        "mask_ratio": args.mask_ratio,
    }