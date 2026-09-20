import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import save_figure, calc_mu_delta_G, get_fittest_neighbor, fermi


def calculate_fitness(population, z, alpha, beta, sigma, T, O, A):
    size = len(population)
    fitnesses = np.zeros(size)
    
    for i in range(size):
        strategy = population[i]
        total_payoff = 0
        neighbors = [x for x in range(size) if x != i]

        for x in neighbors:
            neighbor_strategy = population[x]
            mu_delta_G = calc_mu_delta_G(z)
            delta_G = max(np.random.normal(mu_delta_G, sigma), 0)

            payoff_matrix = {
                'C': {'C': alpha*delta_G - T - 2*O - A, 'D': -T - O},
                'D': {'C': beta*delta_G - O - A, 'D': 0}
            }

            total_payoff += payoff_matrix[strategy][neighbor_strategy]
        
        fitnesses[i] = total_payoff
    
    return fitnesses

def update_population(population, fitnesses, deterministic, K):
    new_population = population.copy()
    size = len(population)

    for i in range(size):
        neighbors = [x for x in range(size) if x != i]

        if deterministic:
            neighbor_pos = get_fittest_neighbor(neighbors, fitnesses)
            if fitnesses[neighbor_pos] > fitnesses[i]:
                new_population[i] = population[neighbor_pos]
        else:
            focal_fitness = fitnesses[i]
            neighbor_pos = random.choice(neighbors)
            neighbor_strategy = population[neighbor_pos]
            neighbor_fitness = fitnesses[neighbor_pos]

            prob = fermi(focal_fitness, neighbor_fitness, K)

            if random.random() < prob:
                new_population[i] = neighbor_strategy

    return new_population

def simulate_population(size=10000, generations=150, initial_cooperator_ratio=0.5, 
                        alpha=1, beta=1, sigma=2, T=200, O=5, A=10, K=0.1, deterministic=False, 
                        save_figures=False, save_data=False, figure_path='fig', data_path='data'):

    population = np.random.choice(['C', 'D'], size=size, p=[initial_cooperator_ratio, 1 - initial_cooperator_ratio])
    history_frequency = []
    history_fitnesses = []
    z = 0

    for gen in range(generations):
        cooperator_count = sum(population == 'C')
        if cooperator_count == 0 or cooperator_count == size:
            history_frequency.append(cooperator_count)
            break

        history_frequency.append(cooperator_count)
        z += cooperator_count / size
        print(size)
        fitnesses = calculate_fitness(population, z, alpha, beta, sigma, T, O, A)
        
        total_fitness = np.sum(fitnesses)
        history_fitnesses.append(total_fitness)

        population = update_population(population, fitnesses, deterministic, K=K)

    if save_figures:
        # Show history frequency
        plt.ylim(0, 100)
        plt.xlim(0, generations)
        plt.plot(np.array(history_frequency) / size * 100, label='Cooperator Frequency')
        plt.xlabel('Generation')
        plt.ylabel('Number of Cooperators')
        plt.title('Cooperator Count Over Generations')
        plt.show()

        path = figure_path
        save_figure(plt, filename=f'{path}_cooperator-history.png')

        # Show history fitness
        plt.xlim(0, generations)
        plt.plot(np.array(history_fitnesses), label='Total fitness')
        plt.xlabel('Generation')
        plt.ylabel('Total fitness')
        plt.title('Total fitness Over Generations')
        plt.show()

        path = figure_path
        save_figure(plt, filename=f'{path}_fitness-history.png')

    if save_data:
        save_data_path = data_path
        data = pd.DataFrame({
            'Cooperator Count': history_frequency,
            'Total fitness': history_fitnesses,
        })

        # Ensure directory exists before writing CSV
        os.makedirs(os.path.dirname(f'{save_data_path}/data.csv'), exist_ok=True)
        data.to_csv(f'{save_data_path}_data.csv', index_label='Generation')

    return population, history_frequency, history_fitnesses


def __main__():
    parser = argparse.ArgumentParser(description='Simulate population dynamics.')
    parser.add_argument('--size', type=int, default=100, help='network size')
    parser.add_argument('--ratio', type=float, default=0.5, help='initial cooperator ratio')
    parser.add_argument('--gen', type=int, default=1000, help='generations')
    parser.add_argument('--alpha', type=float, default=1, help='synergy coefficient')
    parser.add_argument('--beta', type=float, default=1, help='free-riding coefficient')
    parser.add_argument('--sigma', type=float, default=2, help='deta G variance')
    parser.add_argument('--T', type=float, default=200, help='training cost')
    parser.add_argument('--O', type=float, default=5, help='communication cost')
    parser.add_argument('--A', type=float, default=10, help='aggregation cost')
    parser.add_argument('--K', type=float, default=0.1, help='fermi')
    parser.add_argument('--det', action='store_true', help='deterministic')
    parser.add_argument('--seed-start', type=int, default=0, help='Starting seed for simulations')
    parser.add_argument('--seed-end', type=int, default=1, help='Ending seed for simulations')

    args = parser.parse_args()
    seed_start = args.seed_start
    seed_end = args.seed_end

    # Run with different seed
    seeds = range(seed_start, seed_end)
    for seed in tqdm(seeds, desc='Seeds'):
        np.random.seed(seed)
        
        final_population, history_frequency, history_fitness = simulate_population(
            size=args.size, generations=args.gen, initial_cooperator_ratio=args.ratio, alpha=args.alpha, 
            beta=args.beta, sigma=args.sigma, T = args.T, O=args.O, A=args.A, K=args.K, 
            deterministic=args.det, save_figures=False, save_data=False
        )

        res_freq = [float(x) / args.size for x in history_frequency]
        res_fitness = [float(x) for x in history_fitness[1:]]

        # Use strategy and game-type specific directory
        data_dir = f'data/fc/seed_{seed}'

        os.makedirs(data_dir, exist_ok=True)

        mode = "det" if args.det else f"K_{args.K}"
        file_prefix = f'{data_dir}/ratio_{args.ratio}_{mode}_alpha_{args.alpha}_beta_{args.beta}'

        df = pd.DataFrame(res_freq, columns=['Cooperator_Frequency'])
        df.to_csv(f'{file_prefix}_cooperator_frequency.csv', index=False)

        df = pd.DataFrame(res_fitness, columns=['Payoff'])
        df.to_csv(f'{file_prefix}_population_payoff.csv', index=False)


__main__()