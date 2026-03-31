"""
models.py  --  NIDS 3.0

Backbone architectures for the 3-stage supervised contrastive pipeline.

All backbones inherit from ContrastiveBase which provides:
  - Shared projection head (contrastive stage)
  - Shared classifier head (fine-tune stage)
  - freeze/unfreeze utilities required by trainer.py

Available backbones (--backbone flag):
  fttransformer   FT-Transformer (Gorishniy et al., NeurIPS 2021)
  bilstm          Bidirectional LSTM with per-feature embedding
  cnn             1D CNN with batch norm
  cnn_bilstm_se   CNN + SE attention + BiLSTM hybrid

All backbones:
  - Accept input shape (B, 1, F)
  - Produce encoder_dim=256 representations
  - Expose the same encode() / forward() / freeze API
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Base class: shared heads + trainer API
# =============================================================================

class ContrastiveBase(nn.Module):
    """
    Shared projection head, classifier head, and trainer.py interface.
    Subclasses must implement encode(x) -> (B, encoder_dim).
    """
    encoder_dim = 256

    def _build_heads(self, embed_dim, n_classes, dropout):
        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.GELU(),
            nn.Linear(self.encoder_dim, embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def encode(self, x):
        raise NotImplementedError

    def forward(self, x, mode="classify"):
        """
        mode="contrastive" -> L2-normalized projection embeddings
        mode="classify"    -> raw logits
        """
        features = self.encode(x)
        if mode == "contrastive":
            return F.normalize(self.projection(features), dim=1)
        return self.classifier(features)

    # ---- Trainer API ---------------------------------------------------------

    def get_backbone_params(self):
        excluded = set()
        excluded.update(id(p) for p in self.projection.parameters())
        excluded.update(id(p) for p in self.classifier.parameters())
        return [p for p in self.parameters() if id(p) not in excluded]

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_head_params(self):
        return list(self.classifier.parameters())

    def freeze_backbone(self):
        for p in self.get_backbone_params():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.get_backbone_params():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_classifier(self, n_classes, dropout=0.3):
        device = next(self.parameters()).device
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        ).to(device)


# =============================================================================
# Backbone 1: FT-Transformer
# =============================================================================

class ContrastiveFTTransformer(ContrastiveBase):
    """
    Feature-Tokenized Transformer backbone.

    Each of the F input features is projected to a d_model vector via its own
    learned linear map: token_i = x_i * W[i] + b[i]. A learnable [CLS] token
    is prepended; after N pre-norm transformer blocks, the CLS output becomes
    the sequence representation. Self-attention captures arbitrary pairwise
    interactions between all features simultaneously.
    """

    def __init__(
        self,
        n_features=25,
        n_classes=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        attn_dropout=0.1,
        embed_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature linear tokenizer
        self.token_W = nn.Parameter(torch.empty(n_features, d_model))
        self.token_b = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.token_W, a=math.sqrt(5))

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Pre-norm transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.encoder_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.encoder_dim),
            nn.GELU(),
        )
        self._build_heads(embed_dim, n_classes, dropout)

    def encode(self, x):
        B = x.size(0)
        x = x.squeeze(1)                                         # (B, F)
        tokens = x.unsqueeze(-1) * self.token_W + self.token_b  # (B, F, d_model)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)                 # (B, F+1, d_model)
        out = self.transformer(tokens)                           # (B, F+1, d_model)
        return self.encoder_fc(out[:, 0, :])                     # (B, encoder_dim)


# =============================================================================
# Backbone 2: Bidirectional LSTM
# =============================================================================

class ContrastiveBiLSTM(ContrastiveBase):
    """
    Bidirectional LSTM backbone.

    Each feature is treated as a sequence timestep. A per-feature linear
    embedding (1 -> 16) maps scalar values to small vectors before the BiLSTM
    processes the feature sequence. Mean pooling over timesteps produces the
    fixed-length representation.

    Inductive bias: assumes features have useful sequential relationships
    when processed in their original order.
    """

    def __init__(
        self,
        n_features=25,
        n_classes=10,
        embed_dim=128,
        dropout=0.3,
        **kwargs,
    ):
        super().__init__()
        hidden = 128

        self.feature_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.encoder_fc = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, self.encoder_dim),
            nn.GELU(),
        )
        self._build_heads(embed_dim, n_classes, dropout)

    def encode(self, x):
        x = x.squeeze(1).unsqueeze(-1)       # (B, F, 1)
        tokens = self.feature_embed(x)        # (B, F, 16)
        out, _ = self.bilstm(tokens)          # (B, F, 256)
        pooled = out.mean(dim=1)              # (B, 256)
        return self.encoder_fc(pooled)        # (B, encoder_dim)


# =============================================================================
# Backbone 3: 1D CNN
# =============================================================================

class ContrastiveCNN(ContrastiveBase):
    """
    1D Convolutional Neural Network backbone.

    Three convolutional blocks with increasing channel depth (1->64->128->256),
    batch normalization, and GELU activation. Global average pooling collapses
    the spatial dimension to a fixed-size vector.

    Inductive bias: assumes local correlations exist between adjacent features
    in the feature vector.
    """

    def __init__(
        self,
        n_features=25,
        n_classes=10,
        embed_dim=128,
        dropout=0.3,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, self.encoder_dim),
            nn.GELU(),
        )
        self._build_heads(embed_dim, n_classes, dropout)

    def encode(self, x):
        out = self.conv(x)              # (B, 256, F)
        pooled = out.mean(dim=2)        # (B, 256)
        return self.encoder_fc(pooled)  # (B, encoder_dim)


# =============================================================================
# Backbone 4: CNN + SE + BiLSTM hybrid
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1)  # (B, C, 1)
        return x * scale


class ContrastiveCNNBiLSTMSE(ContrastiveBase):
    """
    CNN + Squeeze-and-Excitation + BiLSTM hybrid backbone.

    CNN extracts local feature patterns, SE recalibrates channel importance,
    BiLSTM captures sequential dependencies in the CNN feature maps.
    This is the originally proposed architecture that the ablation study
    evaluates against simpler alternatives.
    """

    def __init__(
        self,
        n_features=25,
        n_classes=10,
        embed_dim=128,
        dropout=0.3,
        **kwargs,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.se = SEBlock(128, reduction=16)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.encoder_fc = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, self.encoder_dim),
            nn.GELU(),
        )
        self._build_heads(embed_dim, n_classes, dropout)

    def encode(self, x):
        cnn_out = self.cnn(x)                    # (B, 128, F)
        se_out = self.se(cnn_out)                # (B, 128, F)
        lstm_in = se_out.permute(0, 2, 1)        # (B, F, 128)
        lstm_out, _ = self.bilstm(lstm_in)       # (B, F, 128)
        pooled = lstm_out.mean(dim=1)            # (B, 128)
        return self.encoder_fc(pooled)           # (B, encoder_dim)


# =============================================================================
# Factory
# =============================================================================

BACKBONE_REGISTRY = {
    "fttransformer": ContrastiveFTTransformer,
    "bilstm":        ContrastiveBiLSTM,
    "cnn":           ContrastiveCNN,
    "cnn_bilstm_se": ContrastiveCNNBiLSTMSE,
}


def build_model(backbone, n_features, n_classes, args):
    """
    Instantiate the correct backbone from the registry.
    FT-Transformer uses its own specific args; others use shared args.
    """
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
        )

    cls = BACKBONE_REGISTRY[backbone]

    if backbone == "fttransformer":
        return cls(
            n_features=n_features,
            n_classes=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            attn_dropout=args.attn_dropout,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        )
    else:
        return cls(
            n_features=n_features,
            n_classes=n_classes,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        )
    