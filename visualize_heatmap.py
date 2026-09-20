import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import save_figure


def heatmap_cooperator_frequency(data_dir, ratio, deterministic, K):
    data_dir = Path(data_dir)
    mode = "det" if deterministic else f"K_{K}"
    records = []
    for seed_dir in sorted(data_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        seed_name = seed_dir.name
        for file_path in seed_dir.glob(
            f"ratio_{ratio}_{mode}_alpha_*_beta_*_cooperator_frequency.csv"
        ):
            match = re.match(
                rf"ratio_{re.escape(str(ratio))}_{re.escape(mode)}"
                r"_alpha_([-+]?\d*\.?\d+)"
                r"_beta_([-+]?\d*\.?\d+)"
                r"_cooperator_frequency\.csv",
                file_path.name
            )

            if match is None:
                print(f"Skip file: {file_path}")
                continue

            alpha = float(match.group(1))
            beta = float(match.group(2))
            df = pd.read_csv(file_path)

            if "Cooperator_Frequency" not in df.columns:
                print(f"Warning: Not found 'Cooperator_Frequency'")
                continue

            if df.empty:
                print(f"Warning: {file_path} is empty")
                continue

            final_frequency = df["Cooperator_Frequency"].iloc[-1]

            records.append({
                "seed": seed_name,
                "alpha": alpha,
                "beta": beta,
                "Cooperator_Frequency": final_frequency
            })

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("Not found")

    mean_results = (
        results
        .groupby(["alpha", "beta"])["Cooperator_Frequency"]
        .mean()
        .reset_index()
    )

    heatmap_data = mean_results.pivot(index="alpha", columns="beta", values="Cooperator_Frequency")
    heatmap_data = heatmap_data.sort_index(ascending=False).sort_index(axis=1)

    return results, mean_results, heatmap_data


def __main__():
    parser = argparse.ArgumentParser(description='Visualize heatmap')
    parser.add_argument('--data_dir', type=str, default="data/lattice", help='data directory')
    parser.add_argument('--save_dir', type=str, default="images", help='data directory')
    parser.add_argument('--ratio', type=float, default=0.5, help='initial cooperator ratio')
    parser.add_argument('--K', type=float, default=0.1, help='fermi')
    parser.add_argument('--det', action='store_true', help='deterministic')
    args = parser.parse_args()

    results, mean_results, heatmap_data = heatmap_cooperator_frequency(
        data_dir=args.data_dir, ratio=args.ratio, deterministic=args.det, K=args.K)

    print("\n=== Raw results ===")
    print(results)

    print("\n=== Mean over seeds ===")
    print(mean_results)

    print("\n=== Heatmap data ===")
    print(heatmap_data)
    

    betas = heatmap_data.columns.astype(float).values
    alphas = heatmap_data.index.astype(float).values

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    im = ax.imshow(
        heatmap_data.to_numpy(),
        cmap="inferno",
        interpolation="bicubic",
        origin="lower",
        extent=[betas.min(), betas.max(), alphas.min(), alphas.max()],
        aspect="auto",
        vmin=0.0,
        vmax=1.0
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    x_ticks = np.arange(round(betas.min(), 1), round(betas.max(), 1) + 0.01, 0.2)
    y_ticks = np.arange(round(alphas.min(), 1), round(alphas.max(), 1) + 0.01, 0.2)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in x_ticks], fontsize=13)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], fontsize=13)

    ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.set_xlabel(r"$\beta$", fontsize=18, fontweight='bold', labelpad=8)
    ax.set_ylabel(r"$\alpha$", fontsize=18, fontweight='bold', labelpad=8)
    
    plt.tight_layout()

    network = args.data_dir.split("/")[1]
    mode = "det" if args.det else f"K_{args.K}"
    save_figure(plt, filename=f'{args.save_dir}/heatmap_{network}_ratio_{args.ratio}_{mode}.png')

__main__()
