# run_augmentation_exp.ps1  --  NIDS 3.0
#
# Reruns FT-Transformer + SupCon WITH feature masking augmentation.
# Directly comparable to exp15 (same backbone, same hyperparams, aug added).
#
# Usage: .\run_augmentation_exp.ps1

Write-Host "Running exp23: FT-Transformer + SupCon + Feature Masking" -ForegroundColor Cyan
Write-Host "Started: $(Get-Date)" -ForegroundColor Cyan

python main.py --train-file data/stratified_Nor100k_Gen50k/UNSW_NB15_training-set.csv --test-file data/stratified_Nor100k_Gen50k/UNSW_NB15_testing-set.csv --run-name exp23_ftt_contrastive_masked --backbone fttransformer --d-model 128 --n-heads 8 --n-layers 4 --contrastive-epochs 150 --contrastive-lr 3e-4 --contrastive-patience 30 --contrastive-min-epochs 50 --n-per-class 16 --temperature 0.15 --aug-mode masking --mask-ratio 0.3 --classifier-epochs 150 --classifier-lr 5e-4 --classifier-patience 25 --classifier-min-epochs 40 --finetune-epochs 150 --finetune-lr 5e-5 --finetune-patience 35 --finetune-min-epochs 60 --batch-size 1024 --no-threshold-calibration

Write-Host "Finished: $(Get-Date)" -ForegroundColor Green
