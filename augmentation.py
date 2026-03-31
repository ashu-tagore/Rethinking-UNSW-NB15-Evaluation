"""
augmentation.py  --  NIDS 3.0

Feature masking augmentation for supervised contrastive pretraining on
tabular network traffic data.

Motivation:
    SupConLoss was designed for vision where augmented views of the same
    image serve as positive pairs, giving the contrastive objective diverse
    signal per anchor. Without augmentation, a Worms anchor (139 training
    samples) sees the same 15 other Worms samples as positives in every
    batch, causing memorization rather than generalization -- confirmed by
    the flat validation contrastive loss in all no-augmentation experiments.

    Feature masking creates two independently corrupted views of each sample
    by randomly zeroing different subsets of features. The model must learn
    representations that are robust to partial feature corruption -- i.e.,
    it must learn WHICH features are consistently discriminative for each
    class rather than memorizing specific feature value combinations.

Usage in contrastive training:
    view1, view2 = create_masked_views(X, mask_ratio=0.3)
    # Stack both views: shape (2B, 1, F)
    X_aug = torch.cat([view1, view2], dim=0)
    y_aug = torch.cat([y, y], dim=0)
    embeddings = model(X_aug, mode="contrastive")
    loss = criterion(embeddings, y_aug)
"""

import torch


def feature_mask(X: torch.Tensor, mask_ratio: float = 0.3) -> torch.Tensor:
    """
    Randomly zero out mask_ratio fraction of features per sample.

    Args:
        X:           Input tensor of shape (B, 1, F)
        mask_ratio:  Fraction of features to zero out (default 0.3 = 30%)

    Returns:
        Masked tensor of same shape as X. Masking is independent per sample.
    """
    B, _, F = X.shape
    n_mask = max(1, int(F * mask_ratio))

    mask = torch.ones(B, F, device=X.device, dtype=X.dtype)
    for i in range(B):
        idx = torch.randperm(F, device=X.device)[:n_mask]
        mask[i, idx] = 0.0

    return X * mask.unsqueeze(1)


def create_masked_views(
    X: torch.Tensor,
    mask_ratio: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create two independently masked views of a batch for contrastive training.

    Each sample gets two different random masks, so the positive pair for
    anchor i is (view1[i], view2[i]) -- two corrupted versions of the same
    network flow record. Both share the same class label, making them valid
    SupConLoss positive pairs.

    Args:
        X:           Input batch, shape (B, 1, F)
        mask_ratio:  Fraction of features to mask per view (default 0.3)

    Returns:
        Tuple (view1, view2), each of shape (B, 1, F)
    """
    view1 = feature_mask(X, mask_ratio)
    view2 = feature_mask(X, mask_ratio)
    return view1, view2


def apply_contrastive_augmentation(
    X: torch.Tensor,
    y: torch.Tensor,
    aug_mode: str = "none",
    mask_ratio: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentation strategy for contrastive pretraining.

    Args:
        X:          Input batch, shape (B, 1, F)
        y:          Labels, shape (B,)
        aug_mode:   "masking" | "none"
        mask_ratio: Feature mask ratio (used when aug_mode="masking")

    Returns:
        (X_out, y_out) where:
            - aug_mode="none":    X_out=X, y_out=y  (no change, shape B)
            - aug_mode="masking": X_out shape (2B, 1, F), y_out shape (2B,)
              Two views stacked; labels repeated to match.
    """
    if aug_mode == "masking":
        view1, view2 = create_masked_views(X, mask_ratio)
        X_out = torch.cat([view1, view2], dim=0)
        y_out = torch.cat([y, y], dim=0)
        return X_out, y_out
    return X, y
