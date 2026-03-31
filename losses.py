import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── Class-Balanced Focal Loss ───────────────────────────────────────────────

class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., CVPR 2019 + Lin et al., ICCV 2017).

    Effective number of samples:  EN(n) = (1 - beta^n) / (1 - beta)
    CB weight for class c:        w_c = (1 - beta) / (1 - beta^n_c)
    Then wrapped with focal modulation: -(1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)   # normalize
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets, mixup_data=None):
        if mixup_data is not None:
            y_a, y_b, lam = mixup_data
            loss_a = self._focal(logits, y_a)
            loss_b = self._focal(logits, y_b)
            return lam * loss_a + (1 - lam) * loss_b
        return self._focal(logits, targets)

    def _focal(self, logits, targets):
        # pt must come from raw softmax — not from weighted CE.
        # Using weighted CE to derive pt corrupts the focal term because
        # CB weights inflate the loss value, making exp(-weighted_ce) ≠ p_t.
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7)

        # focal modulation weight (based on true probability)
        focal_weight = (1.0 - pt) ** self.gamma

        # class-balanced CE applies the CB weights
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")

        return (focal_weight * ce).mean()


# ─── LDAM Loss ───────────────────────────────────────────────────────────────

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019).
    Margin for class j: Delta_j = C / n_j^(1/4)

    Use with Deferred Re-Weighting (DRW):
      - epochs 0..drw_start:  standard CE (no re-weighting)
      - epochs drw_start..:   LDAM + class-frequency re-weighting
    Call set_epoch(epoch) before each epoch so the loss self-adjusts.
    """
    def __init__(self, class_counts, max_m=0.5, s=30, drw_start=20):
        super().__init__()
        self.drw_start = drw_start
        self.s = s
        self.current_epoch = 0

        # margins
        m_list = 1.0 / np.sqrt(np.sqrt(class_counts))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))

        # DRW weights (inverse frequency, normalized)
        freq = class_counts / class_counts.sum()
        weights = 1.0 / freq
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, targets, mixup_data=None):
        use_drw = self.current_epoch >= self.drw_start
        w = self.weights if use_drw else None

        if mixup_data is not None:
            y_a, y_b, lam = mixup_data
            loss_a = self._ldam(logits, y_a, w)
            loss_b = self._ldam(logits, y_b, w)
            return lam * loss_a + (1 - lam) * loss_b

        return self._ldam(logits, targets, w)

    def _ldam(self, logits, targets, weight):
        # subtract margin from the correct class logit
        batch_m = self.m_list[targets].unsqueeze(1)    # (B, 1)
        logits_m = logits.clone()
        logits_m.scatter_add_(
            1,
            targets.unsqueeze(1),
            -batch_m
        )
        return F.cross_entropy(self.s * logits_m, targets, weight=weight)


# ─── Weighted Cross-Entropy (baseline) ───────────────────────────────────────

class WeightedCELoss(nn.Module):
    """Inverse-frequency class-weighted cross-entropy."""
    def __init__(self, class_counts):
        super().__init__()
        freq = class_counts / class_counts.sum()
        weights = 1.0 / freq
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets, mixup_data=None):
        if mixup_data is not None:
            y_a, y_b, lam = mixup_data
            loss_a = F.cross_entropy(logits, y_a, weight=self.weights)
            loss_b = F.cross_entropy(logits, y_b, weight=self.weights)
            return lam * loss_a + (1 - lam) * loss_b
        return F.cross_entropy(logits, targets, weight=self.weights)


# ─── Standard Cross-Entropy (Config A baseline) ──────────────────────────────

class StandardCELoss(nn.Module):
    def forward(self, logits, targets, mixup_data=None):
        if mixup_data is not None:
            y_a, y_b, lam = mixup_data
            loss_a = F.cross_entropy(logits, y_a)
            loss_b = F.cross_entropy(logits, y_b)
            return lam * loss_a + (1 - lam) * loss_b
        return F.cross_entropy(logits, targets)


# ─── Supervised Contrastive Loss ─────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Operates on L2-normalized projection embeddings, NOT logits.
    For each anchor, pulls same-class samples closer and pushes
    different-class samples apart in the embedding space.

    Args:
        temperature: scaling factor for cosine similarities (default 0.07)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, targets):
        """
        Args:
            embeddings: (B, D) L2-normalized projection vectors
            targets:    (B,) integer class labels
        Returns:
            scalar contrastive loss
        """
        device = embeddings.device

        # cosine similarity matrix (B, B), scaled by temperature
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        # mask: same-class pairs (excluding self)
        targets_col = targets.unsqueeze(1)
        positive_mask = torch.eq(targets_col, targets_col.T).float().to(device)
        positive_mask.fill_diagonal_(0)

        # for numerical stability, subtract max from each row
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # denominator: sum over all non-self pairs
        self_mask = torch.ones_like(sim_matrix).fill_diagonal_(0)
        exp_sim = torch.exp(sim_matrix) * self_mask
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # log-prob for each positive pair
        log_prob = sim_matrix - log_denom

        # mean of log-prob over positive pairs per anchor
        n_positives = positive_mask.sum(dim=1)
        safe_mask = (n_positives > 0)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

        loss = -mean_log_prob[safe_mask].mean()
        return loss


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_loss(name, class_counts=None):
    """
    name: "ce" | "weighted_ce" | "cb_focal" | "ldam" | "supcon"
    """
    if name == "ce":
        return StandardCELoss()
    elif name == "weighted_ce":
        return WeightedCELoss(class_counts)
    elif name == "cb_focal":
        return CBFocalLoss(class_counts)
    elif name == "ldam":
        return LDAMLoss(class_counts)
    elif name == "supcon":
        return SupConLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")
    