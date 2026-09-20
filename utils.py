import os
import math
import matplotlib.pyplot as plt


def fermi(focal, neighbor, K=0.3):
    x = (focal - neighbor) / K
    return math.exp(-x) / (1 + math.exp(-x)) if x >= 0 else 1 / (1 + math.exp(x))

def calc_mu_delta_G(z):
    c, eta, theta, s, gamma = 20, 2, 10, 50, -0.7
    return c * eta * theta / ((z / s - gamma) ** (eta + 1))

def get_fittest_neighbor(neighbors, fitnesses):
    best_pos = neighbors[0]
    for pos in neighbors[1:]:
        if fitnesses[pos] > fitnesses[best_pos]:
            best_pos = pos
    return best_pos

def save_figure(plt, filename="figure.png"):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.savefig(filename)
    plt.close()