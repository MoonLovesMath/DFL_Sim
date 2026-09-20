import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import save_figure


def cooperator_frequency(data_dir, alpha, beta, ratio, deterministic, K):
    data_dir = Path(data_dir)
    mode = "det" if deterministic else f"K_{K}"
    all_data = {}
    for seed_dir in sorted(data_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        file_path = (seed_dir/f"ratio_{ratio}_{mode}_alpha_{alpha}_beta_{beta}_cooperator_frequency.csv")
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

    max_generation = 1001#max(len(frequency) for frequency in all_data.values())
    padded_data = []

    for frequency in all_data.values():
        if len(frequency) < max_generation:
            padding = np.full(max_generation - len(frequency), frequency[-1])
            frequency = np.concatenate([frequency, padding])
        padded_data.append(frequency)

    data = np.array(padded_data)
    mean_frequency = np.mean(data, axis=0) * 100
    max_frequency = np.max(data, axis=0) * 100
    min_frequency = np.min(data, axis=0) * 100
    generations = np.arange(max_generation)

    return mean_frequency, max_frequency, min_frequency, generations


def __main__():
    parser = argparse.ArgumentParser(description='Visualize cooperator ratio over generations')
    parser.add_argument('--data_dir', type=str, default="data/lattice", help='data directory')
    parser.add_argument('--save_dir', type=str, default="images", help='data directory')
    parser.add_argument('--ratio', type=float, default=0.5, help='initial cooperator ratio')
    parser.add_argument('--K', type=float, default=0.1, help='fermi')
    parser.add_argument('--det', action='store_true', help='deterministic')
    parser.add_argument('--alpha', type=float, default=1.5, help='synergy coefficient')
    parser.add_argument('--beta', type=float, default=0.5, help='free-riding coefficient')
    args = parser.parse_args()

    mean, max, min, generations = cooperator_frequency(data_dir=args.data_dir, alpha=args.alpha, 
            beta=args.beta, ratio=args.ratio, deterministic=args.det, K=args.K)

    plt.figure(figsize=(7, 4.8), dpi=300)
    plt.fill_between(generations, 0, mean, color='#2ecc71', alpha=1.0, label='Cooperator')
    plt.fill_between(generations, mean, 100, color='#ff7675', alpha=1.0, label='Defector')
    plt.fill_between(generations, min, max, color='#a2b9bc', alpha=0.6)

    max_x = np.max(generations)
    plt.ylim(0, 100)
    plt.xlim(0, max_x)
    step_x = 200
    plt.xticks(range(0, max_x + 1, step_x), fontsize=14)
    plt.yticks(range(0, 101, 20), fontsize=14)

    plt.xlabel('generation', fontsize=16)
    plt.ylabel('frequency', fontsize=16)
    plt.title('Cooperation frequency', fontsize=16, fontweight='bold', pad=10)

    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='lower left', framealpha=0.9, fontsize=11)
    plt.tight_layout()

    network = args.data_dir.split("/")[1]
    mode = "det" if args.det else f"K_{args.K}"
    save_figure(plt, filename=f'{args.save_dir}/line_{network}_ratio_{args.ratio}_{mode}_alpha_{args.alpha}_beta_{args.beta}.png')


__main__()