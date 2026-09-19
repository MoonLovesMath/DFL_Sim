import argparse
from pathlib import Path
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

from utils import save_figure


def heatmap_cooperator_frequency(data_dir):
    data_dir = Path(data_dir)
    records = []
    for seed_dir in sorted(data_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        seed_name = seed_dir.name
        for file_path in seed_dir.glob("alpha_*_beta_*_cooperator_frequency.csv"):
            match = re.match(
                r"alpha_([-+]?\d*\.?\d+)_beta_([-+]?\d*\.?\d+)_cooperator_frequency\.csv",
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
    args = parser.parse_args()

    results, mean_results, heatmap_data = heatmap_cooperator_frequency(args.data_dir)

    print("\n=== Raw results ===")
    print(results)

    print("\n=== Mean over seeds ===")
    print(mean_results)

    print("\n=== Heatmap data ===")
    print(heatmap_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("β", fontsize=16)
    plt.ylabel("α", fontsize=16)
    plt.title("Heatmap Cooperator Frequency")
    plt.tight_layout()

    network = args.data_dir.split("/")[1]
    save_figure(plt, filename=f'{args.save_dir}/heatmap_{network}.png')

__main__()
