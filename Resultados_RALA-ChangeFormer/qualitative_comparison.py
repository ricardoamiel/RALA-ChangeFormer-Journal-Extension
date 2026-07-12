#!/usr/bin/env python3
"""
Qualitative visual comparison: ChangeFormer (V6) vs RALA-ChangeFormer (V7).

Layout per sample row:
  Pre-change (A) | Post-change (B) | Ground Truth | ChangeFormer | RALA-ChangeFormer
The two model columns are error-colored maps (TP/TN/FP/FN) so both the
prediction and its correctness are visible in a single panel.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Paths - Apuntando estrictamente a Resultados_RALA-ChangeFormer
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
RESULTADOS_DIR = BASE if BASE.name == 'Resultados_RALA-ChangeFormer' else BASE / 'Resultados_RALA-ChangeFormer'

DATASETS = {
    'LEVIR-CD+': {
        'pred_v6_dir': RESULTADOS_DIR / 'predictions/V6_LP_test',
        'pred_v7_dir': RESULTADOS_DIR / 'predictions/V7_LP_test',
        'gt_dir':      RESULTADOS_DIR / 'Datasets/LEVIR-CD-PLUS-BENCH/test/label',
        'img_A_dir':   RESULTADOS_DIR / 'Datasets/LEVIR-CD-PLUS-BENCH/test/A',
        'img_B_dir':   RESULTADOS_DIR / 'Datasets/LEVIR-CD-PLUS-BENCH/test/B',
    },
    'SYSU-CD': {
        'pred_v6_dir': RESULTADOS_DIR / 'predictions/V6_SYSU_test',
        'pred_v7_dir': RESULTADOS_DIR / 'predictions/V7_SY_test',
        'gt_dir':      RESULTADOS_DIR / 'Datasets/SYSU-CD-BENCH/test/label',
        'img_A_dir':   RESULTADOS_DIR / 'Datasets/SYSU-CD-BENCH/test/A',
        'img_B_dir':   RESULTADOS_DIR / 'Datasets/SYSU-CD-BENCH/test/B',
    },
    'WHU-CD': {
        'pred_v6_dir': RESULTADOS_DIR / 'predictions/V6_WHU_test',
        'pred_v7_dir': RESULTADOS_DIR / 'predictions/V7_WHU_test',
        'gt_dir':      RESULTADOS_DIR / 'Datasets/WHU-CD-BENCH/test/label',
        'img_A_dir':   RESULTADOS_DIR / 'Datasets/WHU-CD-BENCH/test/A',
        'img_B_dir':   RESULTADOS_DIR / 'Datasets/WHU-CD-BENCH/test/B',
    },
    'S2Looking': {
        'pred_v6_dir': RESULTADOS_DIR / 'predictions/V6_S2Looking_test',
        'pred_v7_dir': RESULTADOS_DIR / 'predictions/V7_S2Looking_test',
        'gt_dir':      RESULTADOS_DIR / 'Datasets/S2Looking-BENCH/test/label',
        'img_A_dir':   RESULTADOS_DIR / 'Datasets/S2Looking-BENCH/test/A',
        'img_B_dir':   RESULTADOS_DIR / 'Datasets/S2Looking-BENCH/test/B',
    },
}

SAMPLES = {
    'LEVIR-CD+': [
        'train_805_0256_0256.png',
        'train_646_0512_0768.png',
        'train_983_0256_0000.png',
        'train_768_0768_0768.png',
    ],
    'SYSU-CD': [
        '02721.png',
        '02370.png',
        '00240.png',
        '03810.png',
    ],
    'WHU-CD': [
        'whucd_03617.png',
        'whucd_05497.png',
        'whucd_06522.png',
        'whucd_07165.png',
    ],
    'S2Looking': [
        '2_0000_0512.png',
        '4991_0000_0256.png',
        '4390_0768_0512.png',
        '3538_0000_0512.png',
    ],
}

# ---------------------------------------------------------------------------
# Matplotlib style & Colormaps
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset':   'stix',
    'axes.titlesize':     8.5,
    'figure.dpi':         150,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02, # Pad mínimo para aprovechar el bounding box de LaTeX
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'text.usetex':        False,
})

CMAP_ERROR = ListedColormap(['#0d0d0d', '#e31a1c', '#1f78b4', '#f7f7f7'])
CMAP_BINARY = ListedColormap(['#0d0d0d', '#f7f7f7'])

COL_TITLES = ['Pre-change (A)', 'Post-change (B)', 'Ground Truth',
              'ChangeFormer', 'RALA-ChangeFormer']

LEGEND_PATCHES = [
    Patch(facecolor='#f7f7f7', edgecolor='#444444', lw=0.4, label='TP (Correct Change)'),
    Patch(facecolor='#0d0d0d', edgecolor='#444444', lw=0.4, label='TN (Correct No-Change)'),
    Patch(facecolor='#e31a1c', edgecolor='#444444', lw=0.4, label='FP (False Alarm)'),
    Patch(facecolor='#1f78b4', edgecolor='#444444', lw=0.4, label='FN (Missed Change)'),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_binary(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    arr = np.array(img)
    if arr.max() == 1:
        return arr.astype(np.uint8)
    return (arr > 127).astype(np.uint8)

def _error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = np.zeros_like(gt, dtype=np.uint8)
    err[(gt == 0) & (pred == 1)] = 1   # FP
    err[(gt == 1) & (pred == 0)] = 2   # FN
    err[(gt == 1) & (pred == 1)] = 3   # TP
    return err

# ---------------------------------------------------------------------------
# Plot Pipeline
# ---------------------------------------------------------------------------
def plot_qualitative(dataset_name: str, paths: dict, samples: list, out_path: Path) -> None:
    n_rows = len(samples)
    n_cols = len(COL_TITLES)

    cell_size = 1.80
    header_h  = 0.75          # Espacio para título + leyenda arriba
    footer_h  = 0.05          # Margen inferior mínimo ya que no hay leyenda abajo

    fig_w = n_cols * cell_size
    fig_h = n_rows * cell_size + header_h + footer_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Ajustamos fracciones del GridSpec dejando el espacio arriba
    top_frac    = 1.0 - (header_h / fig_h)
    bottom_frac = footer_h / fig_h

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.03, wspace=0.03,
        top=top_frac, bottom=bottom_frac,
        left=0.005, right=0.995,
    )

    for row, fname in enumerate(samples):
        img_a   = np.array(Image.open(paths['img_A_dir'] / fname).convert('RGB'))
        img_b   = np.array(Image.open(paths['img_B_dir'] / fname).convert('RGB'))
        gt      = _load_binary(paths['gt_dir']      / fname)
        pred_v6 = _load_binary(paths['pred_v6_dir'] / fname)
        pred_v7 = _load_binary(paths['pred_v7_dir'] / fname)
        emap_v6 = _error_map(gt, pred_v6)
        emap_v7 = _error_map(gt, pred_v7)

        panels = [img_a, img_b, gt, emap_v6, emap_v7]

        for col, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])

            if col < 2:
                ax.imshow(panel, interpolation='lanczos')
            elif col == 2:
                ax.imshow(panel, cmap=CMAP_BINARY, vmin=0, vmax=1, interpolation='nearest')
            else:
                ax.imshow(panel, cmap=CMAP_ERROR, vmin=0, vmax=3, interpolation='nearest')

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            # Títulos de las columnas van en la primera fila de imágenes
            if row == 0:
                ax.set_title(COL_TITLES[col], fontsize=8, pad=4, fontweight='bold')

    # ── Título Superior Principal ───────────────────────────────────────────
    fig.text(0.5, 1.0 - (0.22 / fig_h),
             f'{dataset_name} — Qualitative Comparison',
             ha='center', va='top', fontsize=9.5, fontweight='bold')

    # ── Leyenda Superior (Entre Título y Primera Fila) ──────────────────────
    # bbox_to_anchor posiciona la leyenda justo debajo del título principal
    fig.legend(
        handles=LEGEND_PATCHES, loc='upper center', ncol=4,
        fontsize=7, frameon=False, # Remueve por completo el borde y fondo gris
        bbox_to_anchor=(0.5, 1.0 - (0.42 / fig_h)),
        columnspacing=1.2, handlelength=1.2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    print(f'  [SUCCESS] Vectorized layout saved to: {out_path}')

def main() -> None:
    plots_dir = RESULTADOS_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    for dataset_name, paths in DATASETS.items():
        samples = SAMPLES.get(dataset_name)
        if not samples:
            continue

        missing = []
        for fname in samples:
            for key in ('gt_dir', 'pred_v6_dir', 'pred_v7_dir', 'img_A_dir', 'img_B_dir'):
                p = paths[key] / fname
                if not p.exists():
                    missing.append(str(p))

        if missing:
            print(f'\n[ERROR] Missing files detected for {dataset_name}:')
            for m in missing:
                print(f'  -> NOT FOUND: {m}')
            continue

        safe = dataset_name.replace(' ', '_').replace('+', 'Plus').replace('-', '_')
        out  = plots_dir / f'qualitative_{safe}.pdf'
        print(f'Rendering target: {dataset_name} ({len(samples)} samples)...')
        plot_qualitative(dataset_name, paths, samples, out)

if __name__ == '__main__':
    main()
