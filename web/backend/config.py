from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent.parent  # NIDS 2.0/
RESULTS_DIR = BASE_DIR / "results"

# ── Default experiment ────────────────────────────────────────────────────────
# exp15 = FT-Transformer + Supervised Contrastive Learning (best model, macro F1 0.6504)
# Switch to "exp21" for FT-Transformer direct (macro F1 0.6497, no pretraining overhead)
DEFAULT_EXP = "exp15_ftt_contrastive"

EXP_DIR            = RESULTS_DIR / DEFAULT_EXP
MODEL_PATH         = EXP_DIR / "model.pt"
PREPROCESSING_PATH = EXP_DIR / "preprocessing.pkl"

# ── Attack class ordering (must match label encoder used during training) ─────
CLASSES = [
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

# ── Live capture settings ─────────────────────────────────────────────────────
# Flows idle longer than this are force-finalized and classified
FLOW_TIMEOUT_SEC = 30

# Maximum events kept in the in-memory SSE event log
EVENT_LOG_MAXLEN = 1000
