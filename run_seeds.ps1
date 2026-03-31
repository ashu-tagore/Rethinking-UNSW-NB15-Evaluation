# run_seeds.ps1  --  NIDS 3.0
#
# Runs FT-Transformer Direct and FT-Transformer + SupCon each with 3 seeds.
# Purpose: establish variance estimates so the macro F1 gap between
#          exp21 (direct) and exp15 (contrastive) can be evaluated.
#
# Total: 6 experiments. Estimated time: ~10-12 hours on RTX 3050 Ti.
#
# Usage (from project root in PowerShell):
#   .\run_seeds.ps1
#
# After all 6 finish, run:
#   python analyze_seeds.py --results-dir results/ --prefix seed_

$DATA_TRAIN = "data/stratified_Nor100k_Gen50k/UNSW_NB15_training-set.csv"
$DATA_TEST  = "data/stratified_Nor100k_Gen50k/UNSW_NB15_testing-set.csv"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  3-Seed Variance Estimation Run" -ForegroundColor Cyan
Write-Host "  FT-Transformer: Direct vs + SupCon" -ForegroundColor Cyan
Write-Host "  Started: $(Get-Date)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ── Seed 1: Direct ────────────────────────────────────────────────────────────
Write-Host "`n[1/6] FT-Transformer Direct - Seed 1" -ForegroundColor Yellow
python main.py --train-file $DATA_TRAIN --test-file $DATA_TEST --run-name seed_ftt_direct_s1 --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 0 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration --seed 1
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED - aborting" -ForegroundColor Red; exit 1 }

# ── Seed 1: Contrastive ───────────────────────────────────────────────────────
Write-Host "`n[2/6] FT-Transformer + SupCon - Seed 1" -ForegroundColor Yellow
python main.py --train-file $DATA_TRAIN --test-file $DATA_TEST --run-name seed_ftt_contrastive_s1 --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 150 --contrastive-lr 3e-4 --contrastive-patience 30 --contrastive-min-epochs 50 --n-per-class 16 --temperature 0.15 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration --seed 1
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED - aborting" -ForegroundColor Red; exit 1 }

# ── Seed 2: Direct ────────────────────────────────────────────────────────────
Write-Host "`n[3/6] FT-Transformer Direct - Seed 2" -ForegroundColor Yellow
python main.py --train-file $DATA_TRAIN --test-file $DATA_TEST --run-name seed_ftt_direct_s2 --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 0 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration --seed 2
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED - aborting" -ForegroundColor Red; exit 1 }

# ── Seed 2: Contrastive ───────────────────────────────────────────────────────
Write-Host "`n[4/6] FT-Transformer + SupCon - Seed 2" -ForegroundColor Yellow
python main.py --train-file $DATA_TRAIN --test-file $DATA_TEST --run-name seed_ftt_contrastive_s2 --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 150 --contrastive-lr 3e-4 --contrastive-patience 30 --contrastive-min-epochs 50 --n-per-class 16 --temperature 0.15 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration --seed 2
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED - aborting" -ForegroundColor Red; exit 1 }

# ── Seed 3: Direct ────────────────────────────────────────────────────────────
Write-Host "`n[5/6] FT-Transformer Direct - Seed 3" -ForegroundColor Yellow
python main.py --train-file $DATA_TRAIN --test-file $DATA_TEST --run-name seed_ftt_direct_s3 --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 0 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration --seed 3
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED - aborting" -ForegroundColor Red; exit 1 }

# ── Seed 3: Contrastive ───────────────────────────────────────────────────────
Write-Host "`n[6/6] FT-Transformer + SupCon - Seed 3" -ForegroundColor Yellow
python main.py --train-file $DATA_TRAIN --test-file $DATA_TEST --run-name seed_ftt_contrastive_s3 --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 150 --contrastive-lr 3e-4 --contrastive-patience 30 --contrastive-min-epochs 50 --n-per-class 16 --temperature 0.15 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration --seed 3
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED - aborting" -ForegroundColor Red; exit 1 }

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  All 6 seed runs complete!" -ForegroundColor Green
Write-Host "  Finished: $(Get-Date)" -ForegroundColor Green
Write-Host "  Next: python analyze_seeds.py --results-dir results/ --prefix seed_" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
