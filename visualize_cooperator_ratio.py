import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import save_figure


def cooperator_frequency(data_dir, alpha, beta):
    data_dir = Path(data_dir)
    all_data = {}

    for seed_dir in sorted(data_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        file_path = (seed_dir/f"alpha_{alpha}_beta_{beta}_cooperator_frequency.csv")
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        if "Cooperator_Frequency" not in df.columns:
            continue

        frequency = df["Cooperator_Frequency"].dropna().to_numpy()
        if len(frequency) == 0:
            continue

        all_data[seed_dir.name] = frequency

    if not all_data:
        raise ValueError(f"Not found alpha={alpha}, beta={beta}")

    max_generation = max(len(frequency) for frequency in all_data.values())
    padded_data = []

    for frequency in all_data.values():
        if len(frequency) < max_generation:
            padding = np.full(max_generation - len(frequency), frequency[-1])
            frequency = np.concatenate([frequency, padding])
        padded_data.append(frequency)

    data = np.array(padded_data)
    mean_frequency = np.mean(data, axis=0)
    std_frequency = np.std(data, axis=0)
    generations = np.arange(max_generation)

    return mean_frequency, std_frequency, generations


def __main__():
    parser = argparse.ArgumentParser(description='Visualize cooperator ratio over generations')
    parser.add_argument('--data_dir', type=str, default="data/random", help='data directory')
    parser.add_argument('--alpha', type=float, default=1.5, help='synergy coefficient')
    parser.add_argument('--beta', type=float, default=0.5, help='free-riding coefficient')
    parser.add_argument('--save_dir', type=str, default="images", help='data directory')
    args = parser.parse_args()

    mean, std, generations = cooperator_frequency(data_dir=args.data_dir, alpha=args.alpha, beta=args.beta)

    plt.figure(figsize=(10, 6))
    plt.fill_between(generations, mean - std, mean + std, color="steelblue", alpha=0.25)
    plt.plot(generations, mean, color="steelblue", linewidth=2.5)
    plt.xlabel("Generation")
    plt.ylabel("Cooperator Frequency")
    plt.title(f"Cooperator Frequency (α={args.alpha}, β={args.beta})")
    plt.ylim(0, 1)
    plt.tight_layout()

    network = args.data_dir.split("/")[1]
    save_figure(plt, filename=f'{args.save_dir}/alpha_{args.alpha}_beta_{args.beta}_{network}.png')


__main__()