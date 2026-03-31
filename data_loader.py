import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split as _tts
from sklearn.ensemble import ExtraTreesClassifier
import torch
from torch.utils.data import Dataset, DataLoader


# Official UNSW-NB15 column names (49 columns, no-header raw files only)
COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto",
    "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "service", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]

CATEGORICAL_COLS = ["proto", "state", "service"]

# columns to drop before training (IPs, ports, timestamps, targets)
DROP_COLS = ["srcip", "dstip", "sport", "dsport",
             "stime", "ltime", "id", "attack_cat", "label"]

TARGET_COL = "attack_cat"
N_FEATURES_TO_SELECT = 25

CLASS_NAMES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"
]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _read_csv_auto(path):
    """
    Read one CSV file, auto-detecting whether a header row is present.
    Strips all column name whitespace after loading.
    """
    peek = pd.read_csv(path, nrows=1, header=None, low_memory=False)
    first_cell = str(peek.iloc[0, 0]).strip().lower()
    if first_cell in ("srcip", "id"):
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
    else:
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES, low_memory=False)
    return df


def _clean_labels(df):
    """
    Normalise attack_cat values: strip whitespace, fill blanks with Normal,
    and merge known label variants (Backdoors -> Backdoor).
    Returns df with clean string labels in TARGET_COL.
    """
    df[TARGET_COL] = (
        df[TARGET_COL]
        .fillna("Normal")
        .astype(str)
        .str.strip()
        .replace("", "Normal")
        .replace({"Backdoors": "Backdoor"})  # raw files use both spellings
    )
    return df


def add_interaction_features(df):
    """
    Add interaction features targeting the Analysis/Backdoor/DoS cluster.

    From the confusion matrix, Analysis and Backdoor are predicted as DoS
    50-53% of the time because they share near-identical raw feature values.
    These ratio and interaction features provide discriminative signal that
    raw features alone cannot — e.g. DoS attacks typically have very different
    byte/packet ratios and timing patterns vs Analysis/Backdoor intrusions.

    All divisions use epsilon (1e-6) to avoid divide-by-zero.
    Features are added as new columns; ExtraTrees feature selection
    downstream will automatically drop uninformative ones.

    Called AFTER _encode_and_drop (needs numeric columns) and
    BEFORE _scale_and_select (so ExtraTrees can rank them).
    """
    eps = 1e-6

    def col(name):
        return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

    # ---- Byte ratios (DoS floods bytes asymmetrically vs Analysis) -----------
    df["bytes_ratio"]        = col("sbytes") / (col("dbytes") + eps)
    df["bytes_total"]        = col("sbytes") + col("dbytes")
    df["bytes_per_pkt_src"]  = col("sbytes") / (col("spkts") + eps)
    df["bytes_per_pkt_dst"]  = col("dbytes") / (col("dpkts") + eps)

    # ---- Packet ratios (Analysis has more bidirectional traffic) --------------
    df["pkt_ratio"]          = col("spkts") / (col("dpkts") + eps)
    df["pkt_total"]          = col("spkts") + col("dpkts")

    # ---- Load ratios (DoS has extreme sload asymmetry) -----------------------
    df["load_ratio"]         = col("sload") / (col("dload") + eps)
    df["load_total"]         = col("sload") + col("dload")

    # ---- Timing features (Backdoor has distinct inter-packet timing) ----------
    df["intpkt_ratio"]       = col("sintpkt") / (col("dintpkt") + eps)
    df["dur_per_pkt"]        = col("dur") / (col("spkts") + col("dpkts") + eps)
    df["jit_ratio"]          = col("sjit") / (col("djit") + eps)

    # ---- Mean packet size ratio (Analysis probes have smaller packets) --------
    df["meansz_ratio"]       = col("smeansz") / (col("dmeansz") + eps)

    return df


def _encode_and_drop(df, le=None, cat_encoders=None):
    """
    Encode TARGET_COL with le (fit if not provided), encode categoricals,
    drop non-feature columns, coerce remaining object columns to float.

    Args:
        df:            DataFrame to process
        le:            LabelEncoder for target column (fit on df if None)
        cat_encoders:  dict of {col_name: LabelEncoder} for categorical columns.
                       If None, new encoders are fitted on df (train mode).
                       If provided, they are applied without refitting (test mode).
                       Unseen categories are mapped to a reserved "unknown" class.

    Returns (df_features, y_array, label_encoder, cat_encoders).
    """
    # fit or apply target label encoder
    if le is None:
        le = LabelEncoder()
        le.fit(df[TARGET_COL])
    df[TARGET_COL] = le.transform(df[TARGET_COL])

    # encode categoricals (fit on train, transform on test)
    fitted_new = cat_encoders is None
    if cat_encoders is None:
        cat_encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        cleaned = df[col].fillna("unknown").astype(str).str.strip()

        if fitted_new:
            # train mode: fit encoder, include "unknown" as a catch-all class
            unique_vals = list(cleaned.unique())
            if "unknown" not in unique_vals:
                unique_vals.append("unknown")
            enc = LabelEncoder()
            enc.fit(unique_vals)
            cat_encoders[col] = enc
            df[col] = enc.transform(cleaned)
        else:
            # test mode: map unseen categories to "unknown" before transform
            enc = cat_encoders[col]
            known = set(enc.classes_)
            cleaned = cleaned.where(cleaned.isin(known), other="unknown")
            df[col] = enc.transform(cleaned)

    # drop non-feature columns (TARGET_COL excluded so we can pop it)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns and c != TARGET_COL]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # coerce remaining object columns to numeric
    # (raw files have whitespace-only cells " " in some numeric columns)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip(), errors="coerce"
            )
    df = df.fillna(0)

    y = df.pop(TARGET_COL).values
    return df, y, le, cat_encoders


def _scale_and_select(X_train, X_test, y_train):
    """
    Fit RobustScaler + ExtraTrees feature selector on train, apply to both.
    Returns (X_train_sel, X_test_sel, scaler, top_indices).
    """
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Running feature selection (ExtraTrees)...")
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et.fit(X_train, y_train)
    top_indices = np.sort(
        np.argsort(et.feature_importances_)[::-1][:N_FEATURES_TO_SELECT]
    )
    X_train = X_train[:, top_indices]
    X_test = X_test[:, top_indices]
    print(f"Selected {X_train.shape[1]} features out of {len(et.feature_importances_)}")
    return X_train, X_test, scaler, top_indices


def _print_distribution(y_train, ordered_class_names):
    counts = np.bincount(y_train, minlength=len(ordered_class_names))
    print("\nClass distribution in training set:")
    for i, name in enumerate(ordered_class_names):
        print(f"  {i:2d}  {name:<20s}: {int(counts[i]):>8,}")


# ─── Caching ─────────────────────────────────────────────────────────────────

CACHE_DIR = ".cache"

def _get_cache_path(tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{tag}.pkl")

def _load_cache(tag):
    path = _get_cache_path(tag)
    if os.path.isfile(path):
        print(f"  Loading cached preprocessed data from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def _save_cache(tag, data):
    path = _get_cache_path(tag)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Cached preprocessed data to {path}")


# ─── Public API ──────────────────────────────────────────────────────────────

def load_from_files(train_path, test_path):
    """
    Two-file mode: load pre-split UNSW_NB15_training-set.csv and
    UNSW_NB15_testing-set.csv (both have header rows).

    Scaler and feature selector are fit on train only — no leakage.
    Returns (X_train, X_test, y_train, y_test, label_encoder, ordered_class_names, preprocessing).
    """
    # check cache first
    cache_tag = "split_data"
    cached = _load_cache(cache_tag)
    if cached is not None:
        X_train, X_test, y_train, y_test, preprocessing = cached
        le = preprocessing["label_encoder"]
        ordered_class_names = preprocessing["class_names"]
        _print_distribution(y_train, ordered_class_names)
        return X_train, X_test, y_train, y_test, le, ordered_class_names, preprocessing

    print(f"Loading training file : {train_path}")
    print(f"Loading testing file  : {test_path}")

    train_df = _read_csv_auto(train_path)
    test_df = _read_csv_auto(test_path)
    print(f"Train raw: {len(train_df):,} rows  |  Test raw: {len(test_df):,} rows")

    train_df = _clean_labels(train_df)
    test_df = _clean_labels(test_df)

    # fit LabelEncoder on union of both sets so unseen labels don't crash transform
    all_labels = pd.concat([train_df[TARGET_COL], test_df[TARGET_COL]], ignore_index=True)
    le = LabelEncoder()
    le.fit(all_labels)

    # fit categorical encoders on train only, apply to test (no leakage)
    train_df, y_train, le, cat_encoders = _encode_and_drop(train_df, le=le, cat_encoders=None)
    test_df, y_test, _, _ = _encode_and_drop(test_df, le=le, cat_encoders=cat_encoders)

    # defensive coercion — stratified CSVs saved from raw files may contain
    # whitespace-only cells (e.g. ' ') in numeric columns that survive
    # _encode_and_drop's per-column check when dtype is already numeric
    train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    test_df  = test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_columns = list(train_df.columns)

    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)

    X_train, X_test, scaler, feature_indices = _scale_and_select(X_train, X_test, y_train)

    ordered_class_names = list(le.classes_)

    preprocessing = {
        "scaler": scaler,
        "feature_indices": feature_indices,
        "cat_encoders": cat_encoders,
        "label_encoder": le,
        "class_names": ordered_class_names,
        "feature_columns": feature_columns,
    }

    _save_cache(cache_tag, (X_train, X_test, y_train, y_test, preprocessing))
    _print_distribution(y_train, ordered_class_names)
    return X_train, X_test, y_train, y_test, le, ordered_class_names, preprocessing


def load_and_preprocess(data_dir, test_size=0.20, random_state=42):
    """
    Folder mode: load raw UNSW-NB15_1.csv ... _4.csv from data_dir,
    combine, and create a stratified train/test split internally.

    Categorical encoders are fit on the training split only — no leakage.
    Returns (X_train, X_test, y_train, y_test, label_encoder, ordered_class_names, preprocessing).
    """
    # check cache first
    cache_tag = "raw_data"
    cached = _load_cache(cache_tag)
    if cached is not None:
        X_train, X_test, y_train, y_test, preprocessing = cached
        le = preprocessing["label_encoder"]
        ordered_class_names = preprocessing["class_names"]
        print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
        _print_distribution(y_train, ordered_class_names)
        return X_train, X_test, y_train, y_test, le, ordered_class_names, preprocessing

    csv_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    print(f"Found {len(csv_files)} CSV file(s): {[os.path.basename(f) for f in csv_files]}")

    combined = pd.concat(
        [_read_csv_auto(p) for p in csv_files], axis=0
    ).reset_index(drop=True)
    print(f"Combined dataset: {len(combined):,} rows x {combined.shape[1]} columns")

    combined = _clean_labels(combined)

    # ---- Cap majority classes ------------------------------------------------
    # The raw dataset has 1.77M Normal and 172K Generic samples — 87% and 8.5%
    # of the data respectively. This extreme imbalance (12,000:1 ratio) causes
    # CB Focal Loss to collapse, and makes each epoch take ~6x longer than
    # necessary. We cap Normal and Generic at a fixed maximum while keeping ALL
    # minority class samples intact.
    #
    # Cap values chosen to match the imbalance level of the official split files
    # (where Normal=56K, Generic=40K worked well) scaled up proportionally to
    # give minority classes more training samples from the raw data.
    #
    # Capping is done BEFORE the stratified split so both train and test
    # reflect the same capped distribution.
    CLASS_CAPS = {
        "Normal":  100_000,
        "Generic":  50_000,
    }
    capped_parts = []
    for label, group in combined.groupby(TARGET_COL):
        cap = CLASS_CAPS.get(label, None)
        if cap is not None and len(group) > cap:
            group = group.sample(n=cap, random_state=random_state)
            print(f"  Capped {label}: {len(combined[combined[TARGET_COL]==label]):,} -> {cap:,}")
        capped_parts.append(group)
    combined = pd.concat(capped_parts, axis=0).reset_index(drop=True)
    print(f"Dataset after capping: {len(combined):,} rows")

    # fit target label encoder on all data (need integer labels for stratified split)
    le = LabelEncoder()
    le.fit(combined[TARGET_COL])
    y_all = le.transform(combined[TARGET_COL])

    # stratified split on raw dataframe (before categorical encoding)
    train_df, test_df = _tts(
        combined,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # ---- Save stratified split CSVs for reuse --------------------------------
    # Saved BEFORE encoding so the CSVs contain the original readable labels
    # (e.g. "attack_cat" = "Worms") and can be loaded by load_from_files()
    # in future experiments without needing the raw 4-file dataset again.
    # Folder name includes cap sizes so different caps don't overwrite each other.
    cap_tag = "_".join(f"{k[:3]}{v//1000}k" for k, v in CLASS_CAPS.items())
    stratified_dir = os.path.join(os.path.dirname(data_dir), f"stratified_{cap_tag}")
    os.makedirs(stratified_dir, exist_ok=True)
    train_save_path = os.path.join(stratified_dir, "UNSW_NB15_training-set.csv")
    test_save_path  = os.path.join(stratified_dir, "UNSW_NB15_testing-set.csv")
    if not os.path.isfile(train_save_path):
        train_df.to_csv(train_save_path, index=False)
        test_df.to_csv(test_save_path, index=False)
        print(f"  Stratified split saved to {stratified_dir}/")
    else:
        print(f"  Stratified split already exists at {stratified_dir}/ — skipping save")

    # encode categoricals: fit on train only, apply to test (no leakage)
    train_df, y_train, _, cat_encoders = _encode_and_drop(train_df, le=le, cat_encoders=None)
    test_df, y_test, _, _ = _encode_and_drop(test_df, le=le, cat_encoders=cat_encoders)

    feature_columns = list(train_df.columns)

    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)

    X_train, X_test, scaler, feature_indices = _scale_and_select(X_train, X_test, y_train)

    ordered_class_names = list(le.classes_)

    preprocessing = {
        "scaler": scaler,
        "feature_indices": feature_indices,
        "cat_encoders": cat_encoders,
        "label_encoder": le,
        "class_names": ordered_class_names,
        "feature_columns": feature_columns,
    }

    _save_cache(cache_tag, (X_train, X_test, y_train, y_test, preprocessing))
    _print_distribution(y_train, ordered_class_names)
    return X_train, X_test, y_train, y_test, le, ordered_class_names, preprocessing


# ─── Augmentations ───────────────────────────────────────────────────────────

def gaussian_noise(x, sigma_ratio=0.05):
    return x + torch.randn_like(x) * sigma_ratio


def feature_masking(x, mask_ratio=0.20):
    mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_ratio))
    return x * mask


def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def apply_augmentation(x, y=None, aug_mode="all"):
    """
    aug_mode: "none" | "noise" | "mask" | "mixup" | "all"
    Returns (x_augmented, mixup_data).  mixup_data is None for non-mixup modes.
    """
    mixup_data = None
    if aug_mode == "none":
        return x, mixup_data

    choice = np.random.choice(["noise", "mask", "mixup"]) if aug_mode == "all" else aug_mode

    if choice == "noise":
        x = gaussian_noise(x)
    elif choice == "mask":
        x = feature_masking(x)
    elif choice == "mixup" and y is not None:
        x, y_a, y_b, lam = mixup(x, y)
        mixup_data = (y_a, y_b, lam)

    return x, mixup_data


# ─── Balanced Batch Sampler (for contrastive training) ───────────────────────

class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Samples batches so that every class appears at least `n_per_class` times.

    For contrastive learning, each class needs >= 2 samples per batch to
    form positive pairs.  Minority classes are oversampled (with replacement)
    while majority classes are undersampled per batch.

    Args:
        labels:       1-D array/tensor of integer class labels
        n_per_class:  number of samples to draw per class per batch
        n_batches:    how many batches to yield per epoch (default: auto)
        max_batches:  upper cap on auto-computed n_batches (default: 2000).
                      Prevents absurdly long epochs on large imbalanced datasets
                      where majority class alone would require 200k+ batches.
    """
    def __init__(self, labels, n_per_class=4, n_batches=None, max_batches=2000):
        self.labels = np.asarray(labels)
        self.n_per_class = n_per_class
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        self.batch_size = self.n_classes * self.n_per_class

        # build per-class index lists
        self.class_indices = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

        if n_batches is None:
            # auto: enough batches to see ~all majority-class samples once,
            # but capped to keep epoch duration reasonable
            max_class_size = max(len(idx) for idx in self.class_indices.values())
            auto = max(1, max_class_size // self.n_per_class)
            self.n_batches = min(auto, max_batches)
        else:
            self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                idx = self.class_indices[c]
                # oversample with replacement if class is too small
                replace = len(idx) < self.n_per_class
                chosen = np.random.choice(idx, size=self.n_per_class, replace=replace)
                batch.extend(chosen.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ─── Dataset ─────────────────────────────────────────────────────────────────

class NIDSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, F)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train, X_test, y_train, y_test,
                    batch_size=256, num_workers=0, balanced=False,
                    n_per_class=4):
    """
    Create train and test DataLoaders.

    Args:
        balanced:    if True, use BalancedBatchSampler for contrastive training
                     (ensures every class has n_per_class samples per batch).
                     batch_size is ignored when balanced=True (determined by
                     n_classes * n_per_class).
        n_per_class: samples per class per batch when balanced=True (default 4)
    """
    train_ds = NIDSDataset(X_train, y_train)
    test_ds = NIDSDataset(X_test, y_test)

    if balanced:
        sampler = BalancedBatchSampler(y_train, n_per_class=n_per_class)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=num_workers, pin_memory=(num_workers > 0)
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(num_workers > 0)
        )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )
    return train_loader, test_loader


def get_class_counts(y_train, n_classes=10):
    return np.bincount(y_train, minlength=n_classes).astype(np.float32)
