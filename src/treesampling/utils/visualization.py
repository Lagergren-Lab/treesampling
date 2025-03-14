import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_logweights_heatmap(X: np.ndarray, ax: plt.Axes = None, color_tempering = 100):
    """
    Plot a heatmap of the log-weights of a graph
    :param X: np.ndarray of log-scale weights
    :param ax: plt.Axes object
    :param color_tempering: plots exp(weights/color_tempering) to make the colors less extreme (default 100)
    :return: plt.Axes object
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(10, 8))
        f.tight_layout()
    ax.tick_params(axis='both', labelsize=8)
    weights = X
    sns_plot = sns.heatmap(np.exp(weights / color_tempering),
                           annot=weights,
                           cmap='Blues',
                           fmt='.2f',
                           square=True,
                           ax=ax,
                           annot_kws={"size": 8})
    return ax

