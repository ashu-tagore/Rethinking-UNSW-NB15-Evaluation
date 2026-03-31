# Rethinking UNSW-NB15 Evaluation
### Feature-Tokenized Transformers with Supervised Contrastive Learning for Minority Attack Class Detection

> Bachelor's Thesis — Department of Electronics and Computer Engineering  
> Advanced College of Engineering and Management, Tribhuvan University  
> March 2026

---

## The Problem

A model can achieve **97% accuracy** on UNSW-NB15 while completely failing to detect Worms, Shellcode, and Backdoor attacks. Standard accuracy metrics, dominated by the majority Normal class (~67% of records), hide this failure entirely.

This project confronts that gap directly. We shift the primary evaluation metric to **macro F1** — which weights every class equally regardless of frequency — and build a pipeline that genuinely improves detection of the minority attack classes that matter most in real security operations.

---

## Key Results

Best configuration: **exp15** — FT-Transformer + Supervised Contrastive Learning

| Metric | Value |
|---|---|
| Macro F1 | **0.6504** |
| Overall Accuracy | 86.55% |
| Shellcode F1 | 0.625 |
| Worms F1 | 0.472 |
| Backdoor F1 | 0.197 |

### 4×2 Ablation Study

| Backbone | Parameters | Direct F1 | + SupCon F1 | SupCon Δ |
|---|---|---|---|---|
| **FT-Transformer** (proposed) | 965,770 | 0.6497 | **0.6504** ★ | +0.0007 |
| BiLSTM | 743,978 | 0.6432 | 0.6390 | −0.0042 |
| CNN+BiLSTM+SE (original) | 392,338 | 0.5296 | 0.6133 | +0.0837 |
| CNN | 323,594 | 0.5407 | 0.5373 | −0.0034 |

**Three findings from the ablation:**
1. Architecture selection dominates training strategy — the gap between best and worst backbone (0.120 macro F1) is far larger than the largest training regime effect (0.084)
2. Contrastive benefit is inversely proportional to backbone quality — SupCon rescues weak backbones but adds nothing to strong ones
3. Complexity does not guarantee better performance — the most complex architecture (CNN+BiLSTM+SE) performs worst under direct training

---

## Architecture

The **FT-Transformer** treats each of the 25 UNSW-NB15 flow features as an independent token:

```
25 features → 128-dim token each → prepend CLS token → 26-token sequence
→ 4× Transformer encoder layers (8 attention heads, GELU, pre-norm)
→ CLS output → LayerNorm → 256-dim representation → classifier
```

Self-attention is permutation-invariant and global — no spatial or sequential assumptions are imposed on heterogeneous tabular features. This is the correct inductive bias for network flow data, unlike CNN (assumes local adjacency) or LSTM (assumes sequential order).

### Three-Stage Training Pipeline

```
Stage 0  Supervised Contrastive Pretraining
         Balanced sampler: 16 samples × 10 classes per batch
         SupConLoss (τ = 0.15) — pulls same-class embeddings together

Stage 1  Frozen Classifier Training
         Backbone frozen, classification head trained
         CB Focal Loss (LR 5e-4, patience 25)

Stage 2  Full Fine-Tuning
         All parameters unfrozen
         CB Focal Loss (LR 5e-5, patience 35, min 60 epochs)
```

---

## Dataset

**UNSW-NB15** — produced by the Australian Centre for Cyber Security (ACCS).

| Class | Training | Test |
|---|---|---|
| Normal | 80,000 | 20,000 |
| Generic | 40,000 | 10,000 |
| Exploits | 35,620 | 8,905 |
| Fuzzers | 19,397 | 4,849 |
| DoS | 13,082 | 3,271 |
| Reconnaissance | 11,189 | 2,798 |
| Analysis | 2,142 | 535 |
| Backdoor | 1,863 | 466 |
| Shellcode | 1,209 | 302 |
| Worms | 139 | 35 |

**Imbalance ratio: 575:1** (Normal : Worms)

Download the raw CSVs from the [official UNSW-NB15 page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them under `data/`. The official pre-split files are not used — we build a custom stratified split from the raw CSVs to avoid the known distribution mismatch in the official partition.

---

## Repository Structure

```
NIDS 3.0/
├── data/                        # UNSW-NB15 raw CSVs (not tracked — download separately)
├── results/                     # Per-experiment outputs (model.pt excluded — see .gitignore)
├── venv/                        # Virtual environment (not tracked)
├── web/
│   ├── backend/
│   │   ├── app.py               # FastAPI server + SSE streaming
│   │   ├── capture.py           # Scapy live packet capture
│   │   ├── config.py            # Experiment path config (set DEFAULT_EXP here)
│   │   ├── api.py               # Standalone single/batch prediction API
│   │   └── __init__.py
│   └── frontend/                # HTML/Tailwind/Chart.js (served by FastAPI)
├── .cache/                      # MD5-hashed preprocessing cache (not tracked)
├── .gitignore
├── analyze_seeds.py             # Seed variance analysis across runs
├── augmentation.py              # Feature masking augmentation (exp23)
├── compare_experiments.py       # Ablation comparison visualisations
├── config.py                    # Global hyperparameter config
├── data_loader.py               # Preprocessing pipeline + stratified split
├── inference.py                 # NIDSPredictor class used by web app
├── losses.py                    # CB Focal Loss + SupConLoss implementations
├── main.py                      # Training entry point
├── models.py                    # FT-Transformer, BiLSTM, CNN, CNN+BiLSTM+SE
├── requirements.txt
├── run_augmentation_exp.ps1     # PowerShell: augmentation experiment runner
├── run_seeds.ps1                # PowerShell: overnight seed variance runs
└── trainer.py                   # Training loop (stages 0, 1, 2)
```

---

## Setup

**1. Clone and create a virtual environment**

```powershell
git clone https://github.com/<your-username>/nids-ft-transformer.git
cd "nids-ft-transformer"
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**2. Install PyTorch (CUDA 12.4)**

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**3. Install remaining dependencies**

```powershell
pip install -r requirements.txt
```

---

## Training

```powershell
# FT-Transformer + Supervised Contrastive Learning (reproduces exp15)
python main.py --backbone fttransformer --run-name exp15_ftt_contrastive

# FT-Transformer direct (no contrastive pretraining)
python main.py --backbone fttransformer --mode direct --run-name exp21_ftt_direct

# Full ablation (all 8 configurations)
python main.py --backbone bilstm --mode direct --run-name exp19_bilstm_direct
python main.py --backbone bilstm --run-name exp16_bilstm_contrastive
python main.py --backbone cnn --mode direct --run-name exp20_cnn_direct
python main.py --backbone cnn --run-name exp17_cnn_contrastive
python main.py --backbone cnn_bilstm_se --mode direct --run-name exp22_cnn_bilstm_se_direct
python main.py --backbone cnn_bilstm_se --run-name exp18_cnn_bilstm_se_contrastive
```

Outputs saved to `results/<run-name>/`:
- `model.pt` — best checkpoint
- `preprocessing.pkl` — scaler + label encoders
- `summary.json` — all metrics
- `classification_report.json` — per-class precision/recall/F1
- `*.png` — training curves, confusion matrix, t-SNE

---

## Running the Web Application

```powershell
# From the project root
uvicorn web.backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

| Page | URL | Description |
|---|---|---|
| Home | `/` | Landing page |
| Analyze | `/analyze` | Upload a CSV for batch detection |
| Dashboard | `/dashboard` | Real-time detection statistics |
| History | `/history` | Log of all past analyses |

> **Live capture** (`/api/monitor/start`) requires the terminal to be running as Administrator — Scapy needs raw socket access.

To switch the active model, edit `web/backend/config.py`:
```python
DEFAULT_EXP = "exp15_ftt_contrastive"   # change to any results/ subdirectory
```

---

## Ablation Visualisations

```powershell
# Generate all comparison plots (no GPU needed)
python compare_experiments.py

# Seed variance analysis (after running run_seeds.ps1)
python analyze_seeds.py
```

---

## Limitations

- **No seed variance** — all ablation results are single runs. Small differences (e.g. the 0.0007 FT-Transformer SupCon vs Direct gap) are within run-to-run noise and should not be interpreted causally without mean ± std across multiple seeds.
- **Backdoor detection** remains at 0.197 F1 across all eight configurations. Backdoor traffic is deliberately engineered to mimic legitimate communication — this is a feature-space overlap problem that flow-level statistics alone cannot resolve.
- **Contrastive pretraining without augmentation** does not learn generalizable embedding geometry (validation contrastive loss is flat throughout training). The contrastive stage acts as a warm initialiser rather than achieving its intended representational goal.
- **Single dataset** — findings are validated on UNSW-NB15 only. Cross-dataset validation on CICIDS2017 and CIC-IDS2018 is future work.

---

## Authors

| Name |
|---|
| Ashutosh Bikram Thakur |
| Alish Upreti | 
| Ashim Tiwari |
| Bikash Chaudhary |

**Supervisor:** Prof. Dr. Roshan Chitrakar  
Department of Electronics and Computer Engineering, Nepal College of Information Technology

---

## References

Key references underpinning the methodology:

- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 — FT-Transformer
- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 — SupConLoss
- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019 — CB Focal Loss
- Moustafa & Slay, *UNSW-NB15: A Comprehensive Data Set for NIDS*, MilCIS 2015 — Dataset
- Bahri et al., *SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption*, ICLR 2022 — Tabular augmentation

Full reference list in the thesis report.
