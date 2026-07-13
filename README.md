# RALA-ChangeFormer — Manuscripts and Evaluation Artifacts

Companion repository for the RALA-ChangeFormer research line: efficient
building change detection (BCD) from very high-resolution satellite imagery
by replacing quadratic multi-head self-attention with rank-augmented linear
attention (RALA) in the ChangeFormer encoder.

- Model implementation and training code (conference version):
  <https://github.com/ricardoamiel/RALA-ChangeFormer>
- This repository: manuscripts, evaluation scripts, training logs, metrics,
  and publication figures of the journal extension and thesis.

## Repository structure

| Folder | Content |
|--------|---------|
| `Paper___Ricardo/` | Conference paper (VISAPP 2026) LaTeX sources |
| `Journal_Extension/` | Journal extension LaTeX sources, figures, and plots |
| `Thesis_Ricardo_Amiel/` | Thesis LaTeX sources |
| `Resultados_RALA-ChangeFormer/` | Experiment outputs of the extended benchmarks: per-dataset training/validation curves (`*.npy`), test logs, parsed metrics (`plots/metrics.csv`), and publication figures (`plots/*.pdf`) |

## Evaluation scripts

Both scripts live in `Resultados_RALA-ChangeFormer/` and operate on the
training/test artifacts of the two compared models (V6 = ChangeFormer
baseline, V7 = RALA-ChangeFormer):

- `parse_results.py` — parses the test logs of all eight runs (2 models × 4
  extended datasets), exports `plots/metrics.csv`, and renders the
  training/validation mF1 curves (`plots/learning_curves.pdf`) with both
  models overlaid per dataset.
- `qualitative_comparison.py` — renders the qualitative comparison figures
  (`plots/qualitative_*.pdf`): pre-change (A), post-change (B), ground truth,
  and one TP/TN/FP/FN error-colored map per model.

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
