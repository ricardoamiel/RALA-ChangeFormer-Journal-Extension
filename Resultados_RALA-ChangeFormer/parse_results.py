#!/usr/bin/env python3
"""
Parse ChangeFormerV6 (baseline) and V7 (RALA-ChangeFormer) experiment results
and generate publication-quality PDF plots.

Outputs (in plots/):
  learning_curves.pdf   Training/validation dynamics, CF vs RALA-CF, four datasets
  metrics.csv           Test metrics table for both models on all four datasets

Qualitative figures are handled by qualitative_comparison.py.
"""

import csv
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent

MODELS = ('CF', 'RALA-CF')  # V6 = ChangeFormer baseline, V7 = RALA-ChangeFormer

def _exp(folder: str, log_name: str = 'log_test.txt') -> dict:
    d = BASE / folder
    return {
        'log_test':   d / log_name,
        'scores_npy': d / 'scores_dict.npy',
        'train_acc':  d / 'train_acc.npy',
        'val_acc':    d / 'val_acc.npy',
    }

EXPERIMENTS = {
    'LEVIR-CD+': {
        'CF':      _exp('LEVIR+/V6_LEVIRPLUS'),
        'RALA-CF': _exp('LEVIR+/V7_LP'),
    },
    'SYSU-CD': {
        'CF':      _exp('SYSU/V6_SYSU'),
        'RALA-CF': _exp('SYSU/V7_SY'),
    },
    'WHU-CD': {
        'CF':      _exp('WHU/V6_WHU'),
        'RALA-CF': _exp('WHU/V7_WHU', 'log_test.docx'),
    },
    'S2Looking': {
        'CF':      _exp('S2Looking/V6_S2Looking'),
        'RALA-CF': _exp('S2Looking/V7_S2Looking', 'log_test.docx'),
    },
}

# ---------------------------------------------------------------------------
# Matplotlib style — LaTeX-compatible serif fonts, Type 42 embedding
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset':   'stix',
    'axes.titlesize':     9,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    6.5,
    'lines.linewidth':    1.1,
    'axes.linewidth':     0.7,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'figure.dpi':         150,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'text.usetex':        False,
})

MODEL_COLORS = {
    'CF':      '#2166ac',   # baseline ChangeFormer
    'RALA-CF': '#d73027',   # our approach
}

# ---------------------------------------------------------------------------
# Log parsing — supports .txt and .docx
# ---------------------------------------------------------------------------

_W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def _docx_lines(path: Path) -> list:
    """Extract plain-text paragraph lines from a .docx file via its XML."""
    with zipfile.ZipFile(path) as z:
        xml_bytes = z.read('word/document.xml')
    root = ET.fromstring(xml_bytes)
    lines = []
    for para in root.iter(f'{{{_W_NS}}}p'):
        parts = [r.text for r in para.iter(f'{{{_W_NS}}}t') if r.text]
        line = ''.join(parts)
        if line.strip():
            lines.append(line)
    return lines


def _log_lines(path: Path) -> list:
    """Return text lines from a .txt or .docx log file."""
    if path.suffix.lower() == '.docx':
        return _docx_lines(path)
    return path.read_text(errors='replace').splitlines()


def _parse_kv_line(line: str) -> dict:
    return {k: float(v) for k, v in re.findall(r'([\w_]+):\s*([\d.]+)', line)}


def parse_metrics_from_log(path: Path):
    """Return structured test metrics from the last 'acc:' line in the log."""
    if path is None or not path.exists():
        return None
    lines = _log_lines(path)
    acc_lines = [_parse_kv_line(ln) for ln in lines if ln.strip().startswith('acc:')]
    if not acc_lines:
        return None
    d = acc_lines[-1]
    return {
        'OA':        d.get('acc',         float('nan')),
        'mIoU':      d.get('miou',        float('nan')),
        'mF1':       d.get('mf1',         float('nan')),
        'IoU':       d.get('iou_1',       float('nan')),
        'F1':        d.get('F1_1',        float('nan')),
        'Precision': d.get('precision_1', float('nan')),
        'Recall':    d.get('recall_1',    float('nan')),
    }


def load_metrics(exp: dict) -> dict:
    """Parse log first; fall back to scores_dict.npy if log is missing/empty."""
    m = parse_metrics_from_log(exp.get('log_test'))
    if m is not None:
        return m
    d = np.load(exp['scores_npy'], allow_pickle=True).item()
    return {
        'OA':        d['acc'],
        'mIoU':      d['miou'],
        'mF1':       d['mf1'],
        'IoU':       d['iou_1'],
        'F1':        d['F1_1'],
        'Precision': d['precision_1'],
        'Recall':    d['recall_1'],
    }

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

METRIC_COLS = ['Precision', 'Recall', 'F1', 'IoU', 'OA', 'mF1', 'mIoU']


def save_metrics_csv(experiments: dict, out_path: Path) -> None:
    rows = []
    for name, variants in experiments.items():
        for model in MODELS:
            m = load_metrics(variants[model])
            row = {'Dataset': name, 'Model': model}
            row.update({k: f'{m[k]:.4f}' for k in METRIC_COLS})
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Model'] + METRIC_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {out_path}')

# ---------------------------------------------------------------------------
# Learning curves — CF vs RALA-CF per dataset
# ---------------------------------------------------------------------------

def plot_learning_curves(experiments: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))
    axes = axes.flatten()

    for ax, (name, variants) in zip(axes, experiments.items()):
        ymins = []
        for model in MODELS:
            exp   = variants[model]
            color = MODEL_COLORS[model]
            train_mf1 = np.load(exp['train_acc'])
            val_mf1   = np.load(exp['val_acc'])
            # Some runs log one extra training entry; plot each series on its
            # own epoch axis instead of assuming equal lengths.
            train_ep = np.arange(1, len(train_mf1) + 1)
            val_ep   = np.arange(1, len(val_mf1) + 1)

            best_ep  = int(np.argmax(val_mf1))
            best_val = float(val_mf1[best_ep])
            ymins.append(float(train_mf1.min()))

            ax.plot(train_ep, train_mf1, color=color, lw=0.9, ls='--', alpha=0.65,
                    label=f'{model} train', zorder=2)
            ax.plot(val_ep, val_mf1, color=color, lw=1.15,
                    label=f'{model} val', zorder=3)
            ax.scatter([best_ep + 1], [best_val], color=color, marker='*',
                       s=55, zorder=5,
                       label=f'{model} best val ({best_val:.4f})')

        n_epochs = len(np.load(variants[MODELS[0]]['train_acc']))
        ax.set_title(name, fontweight='bold', pad=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mF1')
        ax.set_xlim(0, n_epochs + 1)
        ax.set_ylim(bottom=max(0.35, min(ymins) - 0.05))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.legend(loc='lower right', framealpha=0.75, edgecolor='#cccccc',
                  ncol=2, columnspacing=0.9, handlelength=1.6)
        ax.grid(True, lw=0.35, alpha=0.5, color='#888888')
        ax.set_axisbelow(True)

    fig.suptitle('ChangeFormer vs RALA-ChangeFormer — Training Dynamics',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    print(f'  Saved: {out_path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    plots_dir = BASE / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print('=== Parsed test metrics ===')
    for name, variants in EXPERIMENTS.items():
        for model in MODELS:
            exp = variants[model]
            m = load_metrics(exp)
            src = 'log' if parse_metrics_from_log(exp.get('log_test')) else 'npy'
            print(
                f'  [{src}] {name:<12s} {model:<8s}: '
                f'Prec={m["Precision"]:.4f}  Rec={m["Recall"]:.4f}  '
                f'F1={m["F1"]:.4f}  IoU={m["IoU"]:.4f}  OA={m["OA"]:.4f}'
            )

    print('\n=== Saving CSV ===')
    save_metrics_csv(EXPERIMENTS, plots_dir / 'metrics.csv')

    print('\n=== Learning curves ===')
    plot_learning_curves(EXPERIMENTS, plots_dir / 'learning_curves.pdf')

    print('\nDone. Run qualitative_comparison.py for visual figures.')


if __name__ == '__main__':
    main()
