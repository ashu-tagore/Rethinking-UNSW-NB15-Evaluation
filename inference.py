"""
inference.py  --  NIDS 3.0

Real-time inference for trained FT-Transformer NIDS models.

Usage (standalone test):
    python inference.py --experiment results/contrastive_fttransformer_d64_L2/

Usage (as module):
    from inference import NIDSPredictor
    predictor = NIDSPredictor("results/contrastive_fttransformer_d64_L2/")
    result = predictor.predict(raw_flow_dict)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch

from models import ContrastiveFTTransformer
from data_loader import add_interaction_features

DROP_COLS = ["srcip", "dstip", "sport", "dsport",
             "stime", "ltime", "id", "attack_cat", "label"]
CATEGORICAL_COLS = ["proto", "state", "service"]


class NIDSPredictor:
    """
    End-to-end predictor: raw network flow dict -> attack classification.

    Loads:
        model.pt           - model weights + thresholds + config
        preprocessing.pkl  - scaler, feature indices, encoders, class names
    """

    def __init__(self, experiment_dir, device=None):
        self.device = self._get_device(device)

        # ---- Load preprocessing ----
        prep_path = os.path.join(experiment_dir, "preprocessing.pkl")
        if not os.path.isfile(prep_path):
            raise FileNotFoundError(f"preprocessing.pkl not found in {experiment_dir}")
        with open(prep_path, "rb") as f:
            prep = pickle.load(f)

        self.scaler = prep["scaler"]
        self.feature_indices = prep["feature_indices"]
        self.cat_encoders = prep["cat_encoders"]
        self.label_encoder = prep["label_encoder"]
        self.class_names = prep["class_names"]
        self.feature_columns = prep["feature_columns"]
        self.n_classes = len(self.class_names)

        # ---- Load checkpoint ----
        ckpt_path = os.path.join(experiment_dir, "model.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"model.pt not found in {experiment_dir}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.thresholds = ckpt.get("thresholds", None)
        cfg = ckpt["config"]
        n_features = ckpt["n_features"]
        n_classes = ckpt["n_classes"]

        # ---- Reconstruct model ----
        self.model = ContrastiveFTTransformer(
            n_features=n_features,
            n_classes=n_classes,
            d_model=cfg.get("d_model", 64),
            n_layers=cfg.get("n_layers", 2),
            n_heads=cfg.get("n_heads", 4),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            embed_dim=cfg.get("embed_dim", 128),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded FT-Transformer from {experiment_dir}")
        print(f"  d_model={cfg.get('d_model', 64)}, "
              f"n_layers={cfg.get('n_layers', 2)}, "
              f"n_heads={cfg.get('n_heads', 4)}")
        print(f"  Classes: {n_classes} | Features: {n_features}")
        print(f"  Thresholds: {'calibrated' if self.thresholds is not None else 'none'}")

    @staticmethod
    def _get_device(override=None):
        if override:
            return torch.device(override)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess(self, raw_input):
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise TypeError(f"Expected dict, list of dicts, or DataFrame")

        df.columns = df.columns.str.strip().str.lower()

        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = "unknown"
            else:
                cleaned = df[col].fillna("unknown").astype(str).str.strip()
                if col in self.cat_encoders:
                    enc = self.cat_encoders[col]
                    known = set(enc.classes_)
                    cleaned = cleaned.where(cleaned.isin(known), other="unknown")
                    df[col] = enc.transform(cleaned)
                else:
                    df[col] = 0

        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip(), errors="coerce"
                )
        df = df.fillna(0)

        # apply same interaction features as training pipeline
        df = add_interaction_features(df)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        X = df.values.astype(np.float32)
        X = self.scaler.transform(X)
        X = X[:, self.feature_indices]

        return torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    @torch.no_grad()
    def predict(self, raw_input):
        single = isinstance(raw_input, dict)
        tensor = self._preprocess(raw_input).to(self.device)

        logits = self.model(tensor, mode="classify")
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        if self.thresholds is not None:
            preds = (probs - self.thresholds).argmax(axis=1)
        else:
            preds = probs.argmax(axis=1)

        results = []
        for i in range(len(preds)):
            pred_idx = preds[i]
            results.append({
                "class": self.class_names[pred_idx],
                "confidence": float(probs[i, pred_idx]),
                "probabilities": {
                    self.class_names[j]: float(probs[i, j])
                    for j in range(self.n_classes)
                },
            })

        return results[0] if single else results

    @torch.no_grad()
    def predict_batch(self, df):
        results = self.predict(df)
        return pd.DataFrame(results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test NIDS 3.0 inference")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    predictor = NIDSPredictor(args.experiment)

    if args.test_csv:
        df = pd.read_csv(args.test_csv, low_memory=False, nrows=args.n_samples)
        df.columns = df.columns.str.strip().str.lower()
        true_labels = df["attack_cat"].fillna("Normal").astype(str).str.strip() \
            if "attack_cat" in df.columns else None

        results = predictor.predict(df)
        print(f"\n{'#':<4} {'Predicted':<18} {'Confidence':<12} {'True':<18}")
        print("-" * 55)
        for i, r in enumerate(results):
            true = true_labels.iloc[i] if true_labels is not None else "?"
            match = "ok" if r["class"] == true else "MISS"
            print(f"{i:<4} {r['class']:<18} {r['confidence']:<12.4f} {true:<18} {match}")
    else:
        demo_flow = {
            "proto": "tcp", "service": "http", "state": "FIN",
            "dur": 0.121478, "sbytes": 100, "dbytes": 6000,
            "sttl": 254, "dttl": 252, "sloss": 0, "dloss": 0,
            "sload": 0, "dload": 6958.18, "spkts": 2, "dpkts": 4,
            "swin": 255, "dwin": 255, "stcpb": 2099014519, "dtcpb": 1809498580,
            "smeansz": 50, "dmeansz": 1500, "trans_depth": 1, "res_bdy_len": 5765,
            "sjit": 0, "djit": 8.75, "sintpkt": 0.02, "dintpkt": 0.008,
            "tcprtt": 0.02, "synack": 0.01, "ackdat": 0.01,
        }
        result = predictor.predict(demo_flow)
        print(f"\n  Predicted: {result['class']}  (confidence: {result['confidence']:.4f})")
        print("\n  All probabilities:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"    {cls:<18} {prob:.4f}  {'#' * int(prob * 40)}")


if __name__ == "__main__":
    main()
    