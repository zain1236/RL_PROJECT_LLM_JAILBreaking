# run_pipeline.ps1
# Runs the full RLTA / RLTA++ reproduction + comparison pipeline end-to-end
# on the small-scale (40-string, 10-epoch) demo configuration.
#
# Prerequisites (see README.md for details):
#   1. Python env with requirements.txt installed (+ Blackwell torch nightly)
#   2. `ollama serve` running, with:
#        ollama pull llama2:7b-chat
#        ollama pull qwen2.5:7b-instruct-q4_K_M   (only needed for the PAIR baseline)
#
# Usage (from the rlta_project root, in PowerShell):
#   .\run_pipeline.ps1

$ErrorActionPreference = "Stop"

# Write-Host "=== Step 0: dataset (AdvBench harmful_strings, 40-string subset) ===" -ForegroundColor Cyan
# python src/data_utils.py --data_dir data --n_total 40 --train_frac 0.8 --seed 42

$common = @(
    "--data_dir", "data/small_40",
    "--agent_model", "EleutherAI/pythia-1.4b",
    "--epochs", "10",
    "--batch_size", "4",
    "--target_backend", "ollama",
    "--ollama_model", "llama2:latest",
    "--bertscore_model_type", "bert-base-uncased",
    "--buffer_size", "64"
)

# Write-Host "`n=== Step 1: train RLTA (BLEU-only baseline) ===" -ForegroundColor Cyan
# python src/train.py --method rlta @common --out_dir outputs/rlta

# Write-Host "`n=== Step 2: train RLTA++ (composite reward + diversity bonus) ===" -ForegroundColor Cyan
# python src/train.py --method rlta_pp @common --out_dir outputs/rlta_pp

# Write-Host "`n=== Step 3: ablation - w/o Rsem (diversity kept, BLEU reward instead of composite) ===" -ForegroundColor Cyan
# python src/train.py --method rlta_pp_no_rsem @common --out_dir outputs/ablation_no_rsem

# Write-Host "`n=== Step 4: ablation - w/o Rdiv (composite reward kept, no diversity bonus) ===" -ForegroundColor Cyan
# python src/train.py --method rlta_pp_no_rdiv @common --out_dir outputs/ablation_no_rdiv

$eval_common = @(
    "--data_dir", "data/small_40",
    "--agent_model", "EleutherAI/pythia-1.4b",
    "--target_backend", "ollama",
    "--ollama_model", "llama2:latest",
    "--bertscore_model_type", "bert-base-uncased"
)

# Write-Host "`n=== Step 5: evaluate all 4 trained variants on the held-out test set ===" -ForegroundColor Cyan
# python src/evaluate.py --checkpoint_dir outputs/rlta --tag "RLTA" @eval_common
# python src/evaluate.py --checkpoint_dir outputs/rlta_pp --tag "RLTA++ (ours)" @eval_common
# python src/evaluate.py --checkpoint_dir outputs/ablation_no_rsem --tag "w/o Rsem" @eval_common
# python src/evaluate.py --checkpoint_dir outputs/ablation_no_rdiv --tag "w/o Rdiv" @eval_common

# Write-Host "`n=== Step 6: PAIR baseline (qwen2.5 as attacker, via Ollama) ===" -ForegroundColor Cyan
# python src/pair_baseline.py --data_dir data/small_40 --target_backend ollama `
#     --target_model llama2:latest --attacker_backend ollama `
#     --attacker_model "qwen2.5:7b-instruct-q4_K_M" --n_iters 3 `
#     --out_file outputs/pair_baseline/eval_result.json

Write-Host "`n=== Step 7: build report assets (Table 2, Table 4, Figure 1, Figure 3) ===" -ForegroundColor Cyan
python src/make_figures_and_tables.py `
    --run "outputs/rlta:RLTA" `
    --run "outputs/rlta_pp:RLTA++ (ours)" `
    --run "outputs/ablation_no_rsem:w/o Rsem" `
    --run "outputs/ablation_no_rdiv:w/o Rdiv" `
    --pair_result outputs/pair_baseline/eval_result.json `
    --out_dir outputs/report_assets

Write-Host "`n=== DONE. Everything you need for the report is in outputs/report_assets/ ===" -ForegroundColor Green
