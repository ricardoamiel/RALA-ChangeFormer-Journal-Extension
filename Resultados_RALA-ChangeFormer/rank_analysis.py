#!/usr/bin/env python3
"""Effective-rank analysis of encoder attention outputs: CF vs RALA-CF.

Measures whether rank-augmented linear attention (RALA) preserves the
representational rank of the token matrices produced by each encoder
attention block, compared to the quadratic MHSA baseline (ChangeFormer).

For every module matching ``Tenc_x2.block{s}.{i}.attn`` a forward hook
captures the output token matrix X (N x C) and computes:

  * erank       -- effective rank, exp(H(sigma / ||sigma||_1))
                   (Roy & Vetterli, EUSIPCO 2007)
  * n_sig1pct   -- number of singular values above 1% of sigma_max
  * stable_rank -- ||X||_F^2 / sigma_max^2
  * erank_ratio -- erank / C (stage-comparable, in [0, 1])

The encoder runs once per temporal image (A, then B), so each tile
contributes two samples per attention block.

This script needs the model class definitions from the RALA-ChangeFormer
repository (https://github.com/ricardoamiel/RALA-ChangeFormer); point
``--repo`` at its root. Checkpoints are the ``best_ckpt.pt`` files already
stored next to the training logs. Inference only, CPU is enough for ~100
tiles.

Usage (run once per model x dataset, then plot):

  python rank_analysis.py run \
      --repo ~/RALA-ChangeFormer \
      --checkpoint WHU/V6_WHU/best_ckpt.pt \
      --model-class ChangeFormerV6 --model-name CF \
      --dataset WHU-CD \
      --data-root Datasets/WHU-CD-BENCH/test \
      --n-tiles 100 --out rank_results

  python rank_analysis.py run \
      --repo ~/RALA-ChangeFormer \
      --checkpoint WHU/V7_WHU/best_ckpt.pt \
      --model-class ChangeFormerV7 --model-name RALA-CF \
      --dataset WHU-CD \
      --data-root Datasets/WHU-CD-BENCH/test \
      --n-tiles 100 --out rank_results

  # repeat for S2Looking (V6_S2Looking / V7_S2Looking,
  #                       Datasets/S2Looking-BENCH/test), then:

  python rank_analysis.py plot --csv rank_results/*.csv --out plots
"""

import argparse
import csv
import glob
import importlib
import inspect
import os
import random
import re
import sys

ATTN_PATTERN = re.compile(r"Tenc_x2\.block(\d)\.(\d+)\.attn$")

MODEL_COLORS = {
    "CF": "#2166ac",
    "RALA-CF": "#d73027",
}


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

def build_model(repo, module_name, class_name, embed_dim):
    """Import the model class from the RALA-ChangeFormer repo and build it,
    passing only the constructor kwargs the class actually accepts."""
    sys.path.insert(0, os.path.abspath(os.path.expanduser(repo)))
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    wanted = {
        "input_nc": 3,
        "output_nc": 2,
        "decoder_softmax": False,
        "embed_dim": embed_dim,
    }
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    dropped = sorted(set(wanted) - set(kwargs))
    if dropped:
        print(f"[build] {class_name} does not accept {dropped}, using its defaults")
    return cls(**kwargs)


def load_checkpoint(model, path, device):
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_G_state_dict", ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as err:
        print(f"[load] strict load failed:\n{err}\n[load] retrying with strict=False")
        incompat = model.load_state_dict(state, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            print(f"[load] missing: {incompat.missing_keys[:10]}")
            print(f"[load] unexpected: {incompat.unexpected_keys[:10]}")
    if "best_epoch_id" in ckpt:
        print(f"[load] best_epoch_id={ckpt['best_epoch_id']}, "
              f"best_val_acc={ckpt.get('best_val_acc'):.4f}")
    return model


def load_tile(path, img_size):
    """PNG -> normalized tensor (1, 3, H, W) in [-1, 1], matching the
    ChangeFormer test-time transform (to_tensor + normalize 0.5/0.5)."""
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    return x.unsqueeze(0)


def spectrum_metrics(x):
    """x: token matrix (N, C). Returns dict of rank statistics."""
    import torch

    sigma = torch.linalg.svdvals(x.float())
    sigma = sigma[sigma > 0]
    p = sigma / sigma.sum()
    erank = torch.exp(-(p * torch.log(p)).sum()).item()
    sigma_max = sigma[0].item()
    n_sig = int((sigma > 0.01 * sigma[0]).sum().item())
    stable = float((sigma ** 2).sum().item() / (sigma_max ** 2))
    return {
        "erank": erank,
        "n_sig1pct": n_sig,
        "stable_rank": stable,
        "sigma_max": sigma_max,
    }


def run(args):
    import torch

    device = torch.device(args.device)
    model = build_model(args.repo, args.module, args.model_class, args.embed_dim)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    # Hook every encoder attention block; captures are drained after each tile.
    captures = []  # (stage, block, tensor)
    hooks = []
    for name, mod in model.named_modules():
        m = ATTN_PATTERN.search(name)
        if not m:
            continue
        stage, block = int(m.group(1)), int(m.group(2))

        def make_hook(stage, block):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captures.append((stage, block, out.detach()))
            return hook

        hooks.append(mod.register_forward_hook(make_hook(stage, block)))
    n_blocks = len(hooks)
    if n_blocks == 0:
        sys.exit("[run] no modules matched Tenc_x2.block*.{i}.attn -- "
                 "check --module/--model-class")
    print(f"[run] hooked {n_blocks} encoder attention blocks")

    list_file = os.path.join(args.data_root, "list", f"{args.split}.txt")
    with open(list_file) as fh:
        names = [line.strip() for line in fh if line.strip()]
    rng = random.Random(args.seed)
    names = rng.sample(names, min(args.n_tiles, len(names)))
    print(f"[run] {len(names)} tiles from {list_file} (seed={args.seed})")

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(
        args.out, f"rank_{args.model_name}_{args.dataset}.csv".replace("/", "-"))
    fields = ["model", "dataset", "tile", "temporal", "stage", "block",
              "N", "C", "erank", "erank_ratio", "n_sig1pct",
              "stable_rank", "sigma_max"]

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for t, tile in enumerate(names):
                xa = load_tile(os.path.join(args.data_root, "A", tile),
                               args.img_size).to(device)
                xb = load_tile(os.path.join(args.data_root, "B", tile),
                               args.img_size).to(device)
                captures.clear()
                model(xa, xb)

                # The encoder forwards image A first, then image B, so the
                # first n_blocks captures belong to A and the rest to B.
                for i, (stage, block, out) in enumerate(captures):
                    temporal = "A" if i < n_blocks else "B"
                    if out.dim() == 4:            # (B, C, H, W) -> (N, C)
                        x = out[0].flatten(1).t()
                    else:                          # (B, N, C)
                        x = out[0]
                    stats = spectrum_metrics(x)
                    writer.writerow({
                        "model": args.model_name,
                        "dataset": args.dataset,
                        "tile": tile,
                        "temporal": temporal,
                        "stage": stage,
                        "block": block,
                        "N": x.shape[0],
                        "C": x.shape[1],
                        "erank": f"{stats['erank']:.4f}",
                        "erank_ratio": f"{stats['erank'] / x.shape[1]:.6f}",
                        "n_sig1pct": stats["n_sig1pct"],
                        "stable_rank": f"{stats['stable_rank']:.4f}",
                        "sigma_max": f"{stats['sigma_max']:.4f}",
                    })
                if (t + 1) % 10 == 0 or t + 1 == len(names):
                    print(f"[run] {t + 1}/{len(names)} tiles")

    for h in hooks:
        h.remove()
    print(f"[run] wrote {out_csv}")


# ---------------------------------------------------------------------------
# plot subcommand
# ---------------------------------------------------------------------------

SUMMARY_METRICS = ["erank", "erank_ratio", "n_sig1pct", "stable_rank"]


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths = sorted(set(sum((glob.glob(p) for p in args.csv), [])))
    if not paths:
        sys.exit(f"[plot] no CSV files matched {args.csv}")
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    print(f"[plot] {len(rows)} samples from {len(paths)} files")

    # group samples by (dataset, model, stage)
    groups = {}
    for r in rows:
        key = (r["dataset"], r["model"], int(r["stage"]))
        groups.setdefault(key, []).append(
            {m: float(r[m]) for m in SUMMARY_METRICS})

    datasets = sorted({k[0] for k in groups})
    seen = {k[1] for k in groups}
    models = [m for m in MODEL_COLORS if m in seen]
    models += sorted(seen - set(models))

    def stats(ds, model, stage, metric):
        vals = [s[metric] for s in groups.get((ds, model, stage), [])]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else None

    fig, axes = plt.subplots(2, len(datasets),
                             figsize=(4.2 * len(datasets), 6.4),
                             squeeze=False)
    for col, ds in enumerate(datasets):
        stages = sorted({k[2] for k in groups if k[0] == ds})
        for metric, row, ylabel in (("erank", 0, "Effective rank"),
                                    ("erank_ratio", 1, "Effective rank / C")):
            ax = axes[row][col]
            for model in models:
                pts = [(st,) + stats(ds, model, st, metric)
                       for st in stages if stats(ds, model, st, metric)]
                if not pts:
                    continue
                xs, means, stds = zip(*pts)
                ax.errorbar(xs, means, yerr=stds, label=model,
                            color=MODEL_COLORS.get(model), marker="o",
                            capsize=3, linewidth=1.6)
            ax.set_xticks(stages)
            ax.set_xlabel("Encoder stage")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(ds)
        axes[0][col].legend()

    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"effective_rank.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[plot] wrote {path}")

    summary_csv = os.path.join(args.out, "effective_rank_summary.csv")
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["dataset", "model", "stage"]
        for m in SUMMARY_METRICS:
            header += [f"{m}_mean", f"{m}_std"]
        writer.writerow(header)
        for ds, model, stage in sorted(groups):
            line = [ds, model, stage]
            for m in SUMMARY_METRICS:
                mean, std = stats(ds, model, stage, m)
                line += [round(mean, 3), round(std, 3)]
            writer.writerow(line)
            print("  ".join(str(v) for v in line))
    print(f"[plot] wrote {summary_csv}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="capture rank statistics for one model")
    p.add_argument("--repo", required=True,
                   help="path to the RALA-ChangeFormer repository root")
    p.add_argument("--checkpoint", required=True, help="path to best_ckpt.pt")
    p.add_argument("--model-class", required=True,
                   help="e.g. ChangeFormerV6 (baseline) or the RALA variant")
    p.add_argument("--module", default="models.ChangeFormer",
                   help="module inside the repo that defines the class")
    p.add_argument("--model-name", required=True, choices=["CF", "RALA-CF"],
                   help="label used in the CSV and plots")
    p.add_argument("--dataset", required=True, help="e.g. WHU-CD, S2Looking")
    p.add_argument("--data-root", required=True,
                   help="split folder containing A/, B/, label/, list/")
    p.add_argument("--split", default="test")
    p.add_argument("--n-tiles", type=int, default=100)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="rank_results")
    p.set_defaults(func=run)

    p = sub.add_parser("plot", help="aggregate CSVs into figure + summary")
    p.add_argument("--csv", nargs="+", required=True,
                   help="CSV files or globs produced by the run subcommand")
    p.add_argument("--out", default="plots")
    p.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
