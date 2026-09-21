# RALA-ChangeFormer — Journal Extension: Evaluation Artifacts

**We have been accepted as position 27 on the Journal CCIC (Communications in Computer and Information Science) !! 🎉🥳 (Monday 7th set)**

Companion repository for the journal extension of RALA-ChangeFormer:
efficient building change detection (BCD) from very high-resolution
satellite imagery by replacing quadratic multi-head self-attention with
rank-augmented linear attention (RALA) in the ChangeFormer encoder.

- Model implementation and training code:
  <https://github.com/ricardoamiel/RALA-ChangeFormer>
- This repository: the extra experiments of the journal extension —
  evaluation and analysis scripts, training logs, parsed metrics, and
  publication figures for the four extended benchmarks.

## Repository structure

Everything lives in `Resultados_RALA-ChangeFormer/`:

| Path | Content |
|------|---------|
| `<DATASET>/V6_*/`, `<DATASET>/V7_*/` | Training/validation curves (`*.npy`) and test logs per run (V6 = ChangeFormer baseline, V7 = RALA-ChangeFormer) |
| `plots/metrics.csv` | Parsed test metrics of all runs |
| `plots/*.pdf` | Publication figures (learning curves, qualitative comparisons) |
| `parse_results.py` | Parses the test logs of all eight runs (2 models × 4 extended datasets), exports `plots/metrics.csv`, and renders the training/validation mF1 curves with both models overlaid per dataset |
| `qualitative_comparison.py` | Renders the qualitative comparison figures: pre-change (A), post-change (B), ground truth, and one TP/TN/FP/FN error-colored map per model |
| `rank_analysis.py` | Effective-rank analysis of the encoder attention outputs (CF vs. RALA-CF) from the existing `best_ckpt.pt` checkpoints: per-stage effective rank, significant singular values, and stable rank, with a plotting subcommand |

Two auxiliary scripts from the conference-stage analysis remain at the
repository root (`DSIFN.py`, `ROCvsPR.py`).

Datasets, model checkpoints (`*.pt`), and raw prediction masks are not
tracked in this repository (see `.gitignore`); the scripts expect them under
`Resultados_RALA-ChangeFormer/Datasets/` and
`Resultados_RALA-ChangeFormer/predictions/` following the layout documented
at the top of each script.

## Extended benchmarks

ChangeFormer and RALA-ChangeFormer were both retrained under an identical
protocol (120 epochs, cosine annealing, initialized from their best LEVIR-CD
checkpoints) on four additional datasets: LEVIR-CD+, SYSU-CD, WHU-CD, and
S2Looking. All experiments ran on an NVIDIA A100 GPU partitioned to 40 GB of
VRAM.

## Citation

If you use this work, please cite the conference paper (RALA-ChangeFormer,
VISAPP 2026) and the journal extension once published.
