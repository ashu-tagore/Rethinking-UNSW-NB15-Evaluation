import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix,
    precision_recall_curve, silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from data_loader import (
    apply_augmentation, CLASS_NAMES, NIDSDataset, BalancedBatchSampler
)
from losses import SupConLoss
from augmentation import apply_contrastive_augmentation

# =============================================================================
# Classification Training Loop
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, aug_mode="none"):
    model.train()
    total_loss = 0.0
    total_samples = 0
    clean_preds, clean_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X, mixup_data = apply_augmentation(X, y, aug_mode)

        optimizer.zero_grad()

        if hasattr(model, "encode"):
            logits = model(X, mode="classify")
        else:
            logits = model(X)

        loss = criterion(logits, y, mixup_data)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

        if mixup_data is None:
            clean_preds.extend(logits.argmax(dim=1).cpu().numpy())
            clean_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples

    if len(clean_targets) > 0:
        acc = accuracy_score(clean_targets, clean_preds)
        macro_f1 = f1_score(clean_targets, clean_preds, average="macro", zero_division=0)
    else:
        acc = float("nan")
        macro_f1 = float("nan")

    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresholds=None):
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if hasattr(model, "encode"):
            logits = model(X, mode="classify")
        else:
            logits = model(X)

        loss = criterion(logits, y, None)
        total_loss += loss.item() * X.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(y.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.array(all_targets)

    if thresholds is not None:
        adjusted = all_probs - thresholds
        preds = adjusted.argmax(axis=1)
    else:
        preds = all_probs.argmax(axis=1)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, preds)
    macro_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1, preds, all_probs, all_targets


# =============================================================================
# Contrastive Training Loop  (now supports feature masking augmentation)
# =============================================================================

def train_contrastive_epoch(model, loader, optimizer, criterion, device,
                             aug_mode="none", mask_ratio=0.3):
    """
    One epoch of supervised contrastive training.
    When aug_mode="masking", each batch is duplicated into two masked views
    before being fed to the model, doubling the effective positive pair count.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Apply contrastive augmentation (doubles batch if masking)
        X_aug, y_aug = apply_contrastive_augmentation(X, y, aug_mode, mask_ratio)

        optimizer.zero_grad()
        embeddings = model(X_aug, mode="contrastive")
        loss = criterion(embeddings, y_aug)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate_contrastive(model, loader, criterion, device,
                          aug_mode="none", mask_ratio=0.3):
    """Evaluate contrastive loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_aug, y_aug = apply_contrastive_augmentation(X, y, aug_mode, mask_ratio)
        embeddings = model(X_aug, mode="contrastive")
        loss = criterion(embeddings, y_aug)
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / total_samples


def train_contrastive_stage(model, train_labels, fit_dataset, val_loader,
                             device, config):
    """
    Supervised contrastive pretraining with optional feature masking.

    Config keys:
        aug_mode (str):    "masking" | "none"  (default "none")
        mask_ratio (float): fraction of features to mask (default 0.3)
    """
    assert len(train_labels) == len(fit_dataset)

    epochs        = config.get("contrastive_epochs", 100)
    lr            = config.get("contrastive_lr", 5e-4)
    weight_decay  = config.get("weight_decay", 1e-4)
    n_per_class   = config.get("n_per_class", 8)
    patience      = config.get("contrastive_patience", 15)
    min_epochs    = config.get("contrastive_min_epochs", 20)
    aug_mode      = config.get("aug_mode", "none")
    mask_ratio    = config.get("mask_ratio", 0.3)

    sampler = BalancedBatchSampler(train_labels, n_per_class=n_per_class)
    con_loader = DataLoader(fit_dataset, batch_sampler=sampler,
                            num_workers=0, pin_memory=False)

    con_val_loader = DataLoader(val_loader.dataset, batch_size=512,
                                shuffle=False, num_workers=0, pin_memory=False)

    params = list(model.get_backbone_params()) + list(model.get_projection_params())
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SupConLoss(temperature=config.get("temperature", 0.07)).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    aug_label = f" [aug={aug_mode}" + (f", mask={mask_ratio}]" if aug_mode == "masking" else "]")
    print(f"  Contrastive batches: {len(sampler)} "
          f"(batch_size={sampler.batch_size}, {n_per_class} per class){aug_label}")

    for epoch in range(epochs):
        tr_loss  = train_contrastive_epoch(
            model, con_loader, optimizer, criterion, device, aug_mode, mask_ratio
        )
        val_loss = evaluate_contrastive(
            model, con_val_loader, criterion, device, aug_mode, mask_ratio
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1

        print_every = min(10, max(1, epochs // 5))
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"  [Contrastive] Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss {tr_loss:.4f} | Val Loss {val_loss:.4f}")

        if epoch + 1 >= min_epochs and no_improve >= patience:
            print(f"  [Contrastive] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, history


# =============================================================================
# Silhouette Score  (measures embedding cluster quality directly)
# =============================================================================

@torch.no_grad()
def compute_silhouette(model, loader, device, max_samples=3000):
    """
    Compute silhouette score on encoder embeddings.

    Silhouette score ranges from -1 to 1:
      - Near 1:  well-separated, compact clusters (good contrastive representations)
      - Near 0:  overlapping clusters
      - Negative: samples are closer to a different cluster's centroid than their own

    Using a subsample of max_samples to keep computation tractable.
    Returns None if fewer than 2 classes are present in the sample.
    """
    model.eval()
    all_emb, all_labels = [], []
    count = 0

    for X, y in loader:
        if count >= max_samples:
            break
        X = X.to(device)
        if hasattr(model, "encode"):
            emb = model.encode(X)
        else:
            return None
        all_emb.append(emb.cpu().numpy())
        all_labels.append(y.numpy())
        count += X.size(0)

    embeddings = np.concatenate(all_emb, axis=0)[:max_samples]
    labels     = np.concatenate(all_labels, axis=0)[:max_samples]

    if len(np.unique(labels)) < 2:
        return None

    try:
        score = silhouette_score(embeddings, labels, metric="cosine",
                                 sample_size=min(max_samples, len(labels)),
                                 random_state=42)
        return float(score)
    except Exception:
        return None


# =============================================================================
# Threshold Optimization
# =============================================================================

def optimize_thresholds(probs, targets):
    n_classes  = probs.shape[1]
    thresholds = np.full(n_classes, 0.5)
    for cls in range(n_classes):
        binary_true = (targets == cls).astype(int)
        if binary_true.sum() == 0:
            continue
        precision, recall, thresh_vals = precision_recall_curve(binary_true, probs[:, cls])
        f1_vals   = np.where(
            (precision + recall) == 0, 0,
            2 * precision * recall / (precision + recall + 1e-8)
        )
        best_idx = np.argmax(f1_vals)
        if best_idx < len(thresh_vals):
            thresholds[cls] = thresh_vals[best_idx]
    return thresholds


# =============================================================================
# Classifier Fine-tune Stage
# =============================================================================

def train_classifier_stage(model, train_loader, val_loader, criterion,
                            device, config, freeze_backbone=True):
    epochs       = config.get("classifier_epochs", 60)
    lr           = config.get("classifier_lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    aug_mode     = config.get("aug_mode", "none")
    patience     = config.get("classifier_patience", 15)
    min_epochs   = config.get("classifier_min_epochs", 15)

    if freeze_backbone and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        params = model.get_head_params()
        print(f"  Backbone frozen, training "
              f"{sum(p.numel() for p in params):,} classifier params")
    else:
        params = model.parameters()

    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
    }
    best_val_f1 = -1
    best_state  = None
    no_improve  = 0

    for epoch in range(epochs):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, aug_mode
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1

        print_every = min(10, max(1, epochs // 5))
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"  [Classifier] Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss {tr_loss:.4f} F1 {tr_f1:.4f} | "
                  f"Val Loss {val_loss:.4f} F1 {val_f1:.4f}")

        if epoch + 1 >= min_epochs and no_improve >= patience:
            print(f"  [Classifier] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    if hasattr(model, "unfreeze_all"):
        model.unfreeze_all()

    return model, history


# =============================================================================
# Legacy Stage Functions
# =============================================================================

def train_stage1(model, train_loader, val_loader, criterion, device, config):
    legacy_config = {
        "classifier_epochs":   config.get("stage1_epochs", 80),
        "classifier_lr":       config.get("lr", 1e-3),
        "weight_decay":        config.get("weight_decay", 1e-4),
        "aug_mode":            config.get("aug_mode", "none"),
        "classifier_patience": config.get("patience", 15),
        "classifier_min_epochs": config.get("min_epochs", 20),
    }
    return train_classifier_stage(model, train_loader, val_loader, criterion,
                                  device, legacy_config, freeze_backbone=False)


def train_stage2(model, train_loader, val_loader, criterion, device, config):
    legacy_config = {
        "classifier_epochs":   config.get("stage2_epochs", 30),
        "classifier_lr":       config.get("stage2_lr", 5e-4),
        "weight_decay":        1e-4,
        "aug_mode":            "none",
        "classifier_patience": config.get("stage2_epochs", 30),
        "classifier_min_epochs": config.get("stage2_epochs", 30),
    }
    return train_classifier_stage(model, train_loader, val_loader, criterion,
                                  device, legacy_config, freeze_backbone=True)


# =============================================================================
# Train/Val Split
# =============================================================================

def _make_train_val_split(train_loader):
    train_dataset = train_loader.dataset
    n_train       = len(train_dataset)
    batch_size    = train_loader.batch_size or 256

    n_classes_in_train = len(np.unique(train_dataset.y.numpy()))
    min_val  = max(n_classes_in_train * 2, 50)
    max_val  = int(0.20 * n_train)
    val_size = int(0.10 * n_train)
    val_size = int(np.clip(val_size, min_val, max_val))
    val_size = min(val_size, n_train - min_val, n_train - 1)
    val_size = max(val_size, 1)

    indices = np.arange(n_train)
    labels  = train_dataset.y.numpy()

    min_class_count = np.bincount(labels).min()
    try:
        if min_class_count >= 2:
            train_idx, val_idx = train_test_split(
                indices, test_size=val_size, random_state=42, stratify=labels
            )
        else:
            raise ValueError("too few samples for stratify")
    except ValueError:
        rng     = np.random.default_rng(42)
        val_idx = rng.choice(indices, size=val_size, replace=False)
        train_idx = np.setdiff1d(indices, val_idx)

    fit_subset  = Subset(train_dataset, train_idx)
    fit_loader  = DataLoader(fit_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_subset  = Subset(train_dataset, val_idx)
    val_loader  = DataLoader(val_subset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    train_labels_fit = labels[train_idx]

    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val samples")
    return fit_loader, val_loader, train_labels_fit, fit_subset


# =============================================================================
# Embedding Extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=5000):
    model.eval()
    all_emb, all_labels = [], []
    count = 0

    for X, y in loader:
        if count >= max_samples:
            break
        X = X.to(device)
        if hasattr(model, "encode"):
            emb = model.encode(X)
        else:
            return None, None
        all_emb.append(emb.cpu().numpy())
        all_labels.append(y.numpy())
        count += X.size(0)

    all_emb    = np.concatenate(all_emb,    axis=0)[:max_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:max_samples]
    return all_emb, all_labels


# =============================================================================
# Full Training Pipeline
# =============================================================================

def run_training(model, train_loader, test_loader, criterion, device,
                 config, output_dir, class_names=None):
    os.makedirs(output_dir, exist_ok=True)
    used_names = class_names if class_names is not None else CLASS_NAMES

    fit_loader, val_loader, train_labels_fit, fit_subset = \
        _make_train_val_split(train_loader)

    use_contrastive    = config.get("contrastive", False)
    con_history        = None
    s2_history         = None
    emb_before         = None
    labels_before      = None
    emb_after          = None
    labels_after       = None
    silhouette_before  = None
    silhouette_after   = None
    silhouette_final   = None

    contrastive_epochs = config.get("contrastive_epochs", 100)

    # ---- Pre-training embeddings + silhouette ----
    if use_contrastive and contrastive_epochs > 0 and hasattr(model, "encode"):
        print("  Capturing pre-training embeddings for t-SNE...")
        emb_before, labels_before = extract_embeddings(model, val_loader, device, 5000)
        silhouette_before = compute_silhouette(model, val_loader, device)
        if silhouette_before is not None:
            print(f"  Silhouette score (before contrastive): {silhouette_before:.4f}")

    # ---- Stage 0: Contrastive pretraining ----
    if use_contrastive and contrastive_epochs > 0:
        print("  === Stage 0: Contrastive Pretraining ===")
        model, con_history = train_contrastive_stage(
            model, train_labels_fit, fit_subset, val_loader, device, config
        )

        print("  Capturing post-contrastive embeddings for t-SNE...")
        emb_after, labels_after = extract_embeddings(model, val_loader, device, 5000)
        silhouette_after = compute_silhouette(model, val_loader, device)
        if silhouette_after is not None:
            print(f"  Silhouette score (after contrastive):  {silhouette_after:.4f}")
    elif use_contrastive:
        print("  [Stage 0 skipped: contrastive_epochs=0]")

    # ---- Stage 1: Classifier training ----
    clf_history = None
    if use_contrastive:
        classifier_epochs        = config.get("classifier_epochs", 60)
        backbone_was_pretrained  = contrastive_epochs > 0
        if classifier_epochs > 0:
            label = "Classifier Fine-tuning (backbone frozen)" if backbone_was_pretrained \
                    else "End-to-End Training (no contrastive pretrain)"
            print(f"  === Stage 1: {label} ===")
            model, clf_history = train_classifier_stage(
                model, fit_loader, val_loader, criterion, device,
                config, freeze_backbone=backbone_was_pretrained
            )
        else:
            print("  [Stage 1 skipped: classifier_epochs=0]")
            clf_history = {k: [] for k in
                           ["train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"]}

        # ---- Stage 2: Full fine-tuning ----
        finetune_epochs = config.get("finetune_epochs", 40)
        if finetune_epochs > 0:
            print("  === Stage 2: Full Fine-tuning (all params) ===")
            ft_config = {
                "classifier_epochs":   finetune_epochs,
                "classifier_lr":       config.get("finetune_lr", 1e-4),
                "weight_decay":        config.get("weight_decay", 1e-4),
                "aug_mode":            "none",
                "classifier_patience": config.get("finetune_patience", 10),
                "classifier_min_epochs": config.get("finetune_min_epochs", 10),
            }
            model, ft_history = train_classifier_stage(
                model, fit_loader, val_loader, criterion, device,
                ft_config, freeze_backbone=False
            )
            s2_history = ft_history
    else:
        print("  --- Stage 1: Full model training ---")
        model, clf_history = train_stage1(model, fit_loader, val_loader,
                                          criterion, device, config)
        if config.get("two_stage", False):
            print("  --- Stage 2: Classifier head retraining ---")
            model, s2_history = train_stage2(model, fit_loader, val_loader,
                                             criterion, device, config)

    # ---- Silhouette after full fine-tuning ----
    if hasattr(model, "encode"):
        silhouette_final = compute_silhouette(model, val_loader, device)
        if silhouette_final is not None:
            print(f"  Silhouette score (after fine-tuning):  {silhouette_final:.4f}")

    # ---- Evaluation ----
    _, acc, macro_f1, preds, probs, targets = evaluate(
        model, test_loader, criterion, device, thresholds=None
    )
    print(f"\n  [Before calibration] Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")

    thresholds = None
    use_thresh = config.get("threshold_calibration", True if use_contrastive else False)
    if use_thresh:
        _, _, _, _, val_probs, val_targets = evaluate(
            model, val_loader, criterion, device, thresholds=None
        )
        thresholds = optimize_thresholds(val_probs, val_targets)
        _, acc_cal, macro_f1_cal, preds_cal, _, _ = evaluate(
            model, test_loader, criterion, device, thresholds=thresholds
        )
        print(f"  [After calibration]  Acc: {acc_cal:.4f} | Macro F1: {macro_f1_cal:.4f}")
        preds    = preds_cal
        acc      = acc_cal
        macro_f1 = macro_f1_cal

    # ---- Classification report ----
    unique_labels          = sorted(np.unique(targets))
    target_names_for_report = [used_names[i] for i in unique_labels if i < len(used_names)]

    report = classification_report(
        targets, preds,
        labels=unique_labels,
        target_names=target_names_for_report,
        zero_division=0, output_dict=True
    )

    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Classification report saved to {report_path}")

    results = {
        "accuracy":             acc,
        "macro_f1":             macro_f1,
        "per_class_f1":         {name: report[name]["f1-score"]
                                  for name in target_names_for_report if name in report},
        "preds":                preds,
        "probs":                probs,
        "targets":              targets,
        "thresholds":           thresholds,
        "contrastive_history":  con_history,
        "stage1_history":       clf_history,
        "stage2_history":       s2_history,
        "silhouette_before":    silhouette_before,
        "silhouette_after_contrastive": silhouette_after,
        "silhouette_final":     silhouette_final,
    }

    print("\n" + classification_report(
        targets, preds,
        labels=unique_labels,
        target_names=target_names_for_report,
        zero_division=0
    ))

    # ---- Plots ----
    plot_training_curves_separate(clf_history, s2_history, output_dir, con_history)
    plot_training_curves(clf_history, s2_history, output_dir, con_history)  # keep combined
    plot_confusion_matrix(targets, preds, used_names, output_dir)
    plot_per_class_f1(report, target_names_for_report, output_dir)
    plot_precision_recall_curves(probs, targets, target_names_for_report, output_dir)
    plot_class_imbalance(train_loader.dataset.y.numpy(), used_names, output_dir)

    if con_history:
        plot_contrastive_loss(con_history, output_dir)

    # Save silhouette scores to a JSON summary
    sil_path = os.path.join(output_dir, "silhouette_scores.json")
    with open(sil_path, "w") as f:
        json.dump({
            "before_contrastive": silhouette_before,
            "after_contrastive":  silhouette_after,
            "after_finetuning":   silhouette_final,
        }, f, indent=2)

    if emb_before is not None and emb_after is not None:
        plot_tsne_comparison(emb_before, labels_before, emb_after, labels_after,
                             used_names, output_dir)
    elif hasattr(model, "encode"):
        emb_final, labels_final = extract_embeddings(model, val_loader, device, 5000)
        if emb_final is not None:
            plot_tsne_single(emb_final, labels_final, used_names, output_dir,
                             title="Learned Embeddings (Direct Training)")

    return results


# =============================================================================
# Plotting — Separate High-Quality Training Graphs
# =============================================================================

STAGE_COLORS = {
    "contrastive": ("#9467bd", "#c5b0d5"),
    "classifier":  ("#1f77b4", "#aec7e8"),
    "finetune":    ("#2ca02c", "#98df8a"),
}
SMOOTH_W = 0.85


def _smooth(values, weight=SMOOTH_W):
    if not values:
        return values
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def _plot_stage(ax, values, label, color, light_color, raw_alpha=0.15):
    """Plot one stage's metric on a given axis with reset x-axis."""
    if not values:
        return
    x = list(range(len(values)))
    ax.plot(x, values, color=color, alpha=raw_alpha, linewidth=1.0)
    ax.plot(x, _smooth(values), color=color, linewidth=2.2, label=label)


def _style_ax(ax, title, ylabel, xlabel="Epoch"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


def plot_training_curves_separate(s1_history, s2_history, output_dir, con_history=None):
    """
    Generate THREE separate high-quality figures:
        training_loss.png     — loss across all three stages
        training_accuracy.png — accuracy for stages 1 & 2
        training_f1.png       — macro F1 for stages 1 & 2

    Each figure has one subplot per stage, with the x-axis reset to 0
    at the start of each stage. Stage boundaries are labeled clearly.
    """
    stages = []
    if con_history and con_history.get("train_loss"):
        stages.append(("Stage 0\nContrastive Pretraining", "contrastive", con_history))
    if s1_history and s1_history.get("train_loss"):
        stages.append(("Stage 1\nFrozen Classifier", "classifier", s1_history))
    if s2_history and s2_history.get("train_loss"):
        stages.append(("Stage 2\nFull Fine-Tuning", "finetune", s2_history))

    if not stages:
        return

    n_stages = len(stages)

    # ── Loss ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 5), squeeze=False)
    fig.suptitle("Training Loss — All Stages", fontsize=15, fontweight="bold", y=1.02)
    axes = axes[0]

    for i, (stage_name, stage_key, hist) in enumerate(stages):
        tc, vc = STAGE_COLORS[stage_key]
        _plot_stage(axes[i], hist.get("train_loss", []), "Train", tc, vc)
        _plot_stage(axes[i], hist.get("val_loss",   []), "Val",   vc, tc)
        _style_ax(axes[i], stage_name, "Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Accuracy ──────────────────────────────────────────────────────────────
    clf_stages = [(n, k, h) for n, k, h in stages if k in ("classifier", "finetune")]
    if clf_stages:
        fig, axes = plt.subplots(1, len(clf_stages), figsize=(6 * len(clf_stages), 5),
                                 squeeze=False)
        fig.suptitle("Training Accuracy — Classification Stages",
                     fontsize=15, fontweight="bold", y=1.02)
        axes = axes[0]

        for i, (stage_name, stage_key, hist) in enumerate(clf_stages):
            tc, vc = STAGE_COLORS[stage_key]
            _plot_stage(axes[i], hist.get("train_acc", []), "Train", tc, vc)
            _plot_stage(axes[i], hist.get("val_acc",   []), "Val",   vc, tc)
            _style_ax(axes[i], stage_name, "Accuracy")
            axes[i].set_ylim(0.5, 1.02)
            axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_accuracy.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

        # ── Macro F1 ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(clf_stages), figsize=(6 * len(clf_stages), 5),
                                 squeeze=False)
        fig.suptitle("Training Macro F1 — Classification Stages",
                     fontsize=15, fontweight="bold", y=1.02)
        axes = axes[0]

        for i, (stage_name, stage_key, hist) in enumerate(clf_stages):
            tc, vc = STAGE_COLORS[stage_key]
            _plot_stage(axes[i], hist.get("train_f1", []), "Train", tc, vc)
            _plot_stage(axes[i], hist.get("val_f1",   []), "Val",   vc, tc)
            _style_ax(axes[i], stage_name, "Macro F1")
            axes[i].set_ylim(0.3, 1.02)
            axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_f1.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()


# =============================================================================
# Plotting — Combined Training Curves (backward compat)
# =============================================================================

def plot_training_curves(s1_history, s2_history, output_dir, con_history=None):
    """Combined 3-panel plot (legacy). Kept for backward compatibility."""
    TRAIN_COLOR = "#1f77b4"
    VAL_COLOR   = "#ff7f0e"
    CON_COLOR   = "#9467bd"

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.patch.set_facecolor("white")

    def _plot_metric(ax, train_vals, val_vals, x_offset=0,
                     train_color=TRAIN_COLOR, val_color=VAL_COLOR,
                     train_label="Training", val_label="Validation"):
        if not train_vals:
            return
        x = list(range(x_offset, x_offset + len(train_vals)))
        ax.plot(x, train_vals, color=train_color, alpha=0.15, linewidth=1.0)
        ax.plot(x, val_vals,   color=val_color,   alpha=0.15, linewidth=1.0)
        ax.plot(x, _smooth(train_vals), color=train_color, linewidth=2.0, label=train_label)
        ax.plot(x, _smooth(val_vals),   color=val_color,   linewidth=2.0, label=val_label)

    offset = 0

    if con_history and con_history.get("train_loss"):
        n_con = len(con_history["train_loss"])
        _plot_metric(axes[0], con_history["train_loss"], con_history["val_loss"],
                     x_offset=0, train_color=CON_COLOR, val_color="#c5b0d5",
                     train_label="Contrastive Train", val_label="Contrastive Val")
        for ax in axes:
            ax.axvline(n_con - 0.5, color="#aaaaaa", linestyle="--", linewidth=1.0, alpha=0.7)
        offset = n_con

    if s1_history and s1_history.get("train_loss"):
        n1 = len(s1_history["train_loss"])
        _plot_metric(axes[0], s1_history["train_loss"], s1_history["val_loss"],
                     x_offset=offset)
        _plot_metric(axes[1], s1_history.get("train_acc",[]), s1_history.get("val_acc",[]),
                     x_offset=offset, train_label="Classifier Train", val_label="Classifier Val")
        _plot_metric(axes[2], s1_history.get("train_f1",[]),  s1_history.get("val_f1",[]),
                     x_offset=offset, train_label="Classifier Train", val_label="Classifier Val")
        if s2_history and s2_history.get("train_loss"):
            for ax in axes:
                ax.axvline(offset + n1 - 0.5, color="#aaaaaa", linestyle="--",
                           linewidth=1.0, alpha=0.7)
        offset += n1

    if s2_history and s2_history.get("train_loss"):
        _plot_metric(axes[0], s2_history["train_loss"], s2_history["val_loss"],
                     x_offset=offset, train_color="#2ca02c", val_color="#98df8a",
                     train_label="Fine-tune Train", val_label="Fine-tune Val")
        _plot_metric(axes[1], s2_history.get("train_acc",[]), s2_history.get("val_acc",[]),
                     x_offset=offset, train_color="#2ca02c", val_color="#98df8a",
                     train_label="Fine-tune Train", val_label="Fine-tune Val")
        _plot_metric(axes[2], s2_history.get("train_f1",[]),  s2_history.get("val_f1",[]),
                     x_offset=offset, train_color="#2ca02c", val_color="#98df8a",
                     train_label="Fine-tune Train", val_label="Fine-tune Val")

    titles  = ["Loss", "Accuracy", "Macro F1"]
    ylabels = ["Loss", "Accuracy", "Macro F1"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabels[i], fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.tick_params(labelsize=9)
        ax.grid(False)

    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Plotting — Other Figures
# =============================================================================

def plot_contrastive_loss(con_history, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(con_history["train_loss"], label="Train Contrastive Loss", linewidth=2)
    ax.plot(con_history["val_loss"],   label="Val Contrastive Loss",   linewidth=2)
    ax.set_title("Contrastive Pretraining Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("SupCon Loss", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contrastive_loss.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(targets, preds, class_names, output_dir):
    unique_labels = sorted(np.unique(np.concatenate([targets, preds])))
    used_names    = [class_names[i] for i in unique_labels if i < len(class_names)]
    cm = confusion_matrix(targets, preds, labels=unique_labels, normalize="true")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=used_names, yticklabels=used_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_per_class_f1(report, class_names, output_dir):
    classes   = [c for c in class_names if c in report]
    f1_scores = [report[c]["f1-score"] for c in classes]
    colors    = ["#e74c3c" if f < 0.5 else "#f39c12" if f < 0.75 else "#2ecc71"
                 for f in f1_scores]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color="gray", linestyle="--", alpha=0.5, label="0.75 threshold")
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.legend()

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150)
    plt.close()


def _run_tsne(embeddings, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, init="pca", learning_rate="auto")
    return tsne.fit_transform(embeddings)


def _plot_tsne_ax(ax, coords_2d, labels, class_names, title):
    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[cmap(i)],
                   label=name, s=8, alpha=0.6, edgecolors="none")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", markerscale=2, framealpha=0.8)


def plot_tsne_comparison(emb_before, labels_before, emb_after, labels_after,
                         class_names, output_dir):
    print("  Generating t-SNE visualization (this may take a minute)...")
    coords_before = _run_tsne(emb_before)
    coords_after  = _run_tsne(emb_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _plot_tsne_ax(ax1, coords_before, labels_before, class_names,
                  "Before Contrastive Pretraining\n(Random Initialization)")
    _plot_tsne_ax(ax2, coords_after,  labels_after,  class_names,
                  "After Contrastive Pretraining\n(Learned Embeddings)")
    fig.suptitle("t-SNE Visualization of Encoder Embeddings", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  t-SNE plot saved.")


def plot_tsne_single(embeddings, labels, class_names, output_dir, title=None):
    print("  Generating t-SNE visualization (this may take a minute)...")
    coords = _run_tsne(embeddings)
    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_tsne_ax(ax, coords, labels, class_names, title or "Learned Embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_embeddings.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  t-SNE plot saved.")


def plot_precision_recall_curves(probs, targets, class_names, output_dir):
    n_classes    = len(class_names)
    cols         = min(5, n_classes)
    rows         = (n_classes + cols - 1) // cols
    fig, axes    = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_flat    = np.array(axes).flatten()
    unique_labels = sorted(np.unique(targets))

    for i, cls in enumerate(unique_labels):
        if i >= len(class_names):
            break
        ax = axes_flat[i]
        binary_true = (targets == cls).astype(int)
        if binary_true.sum() == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center")
            ax.set_title(class_names[i], fontsize=10)
            continue
        precision, recall, _ = precision_recall_curve(binary_true, probs[:, cls])
        auc_pr = -np.trapz(precision, recall)
        ax.plot(recall, precision, color="#2E86C1", linewidth=1.5)
        ax.fill_between(recall, precision, alpha=0.15, color="#2E86C1")
        ax.set_title(f"{class_names[i]} (AUC={auc_pr:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    for j in range(len(unique_labels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Per-Class Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=150)
    plt.close()


def plot_class_imbalance(train_labels, class_names, output_dir):
    counts = np.bincount(train_labels, minlength=len(class_names))
    names  = [class_names[i] if i < len(class_names) else f"Class {i}"
              for i in range(len(counts))]
    order  = np.argsort(counts)[::-1]
    sorted_names  = [names[i] for i in order]
    sorted_counts = counts[order]

    max_count = sorted_counts[0]
    colors    = []
    for c in sorted_counts:
        ratio = np.log10(c + 1) / np.log10(max_count + 1)
        r = int(220 * (1 - ratio) + 44 * ratio)
        g = int(60  * (1 - ratio) + 62 * ratio)
        b = int(60  * (1 - ratio) + 80 * ratio)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sorted_names)), sorted_counts, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Number of Samples (log scale)", fontsize=11)
    ax.set_title("UNSW-NB15 Training Set Class Distribution", fontsize=13,
                 fontweight="bold")

    for bar, count in zip(bars, sorted_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count * 1.15,
                f"{count:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    imbalance_ratio = int(sorted_counts[0] / sorted_counts[-1])
    ax.text(0.98, 0.95, f"Imbalance ratio: {imbalance_ratio:,}:1",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            fontweight="bold", color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FADBD8",
                      edgecolor="#C0392B", alpha=0.9))

    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_imbalance.png"), dpi=150)
    plt.close()