import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_logweights_heatmap(G: nx.DiGraph, ax: plt.Axes = None, color_tempering = 100):
    """
    Plot a heatmap of the log-weights of a graph
    :param G: nx.DiGraph object
    :param ax: plt.Axes object
    :param color_tempering: plots exp(weights/color_tempering) to make the colors less extreme (default 100)
    :return: plt.Axes object
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(10, 8))
        f.set_tight_layout()
    ax.tick_params(axis='both', labelsize=8)
    weights = nx.to_numpy_array(G)
    sns_plot = sns.heatmap(np.exp(weights / color_tempering),
                           annot=weights,
                           cmap='Blues',
                           fmt='.2f',
                           square=True)
    return ax

