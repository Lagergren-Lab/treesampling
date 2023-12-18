import random
import time

import networkx as nx


def random_graph(n_nodes) -> nx.DiGraph:
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    for u, v in graph.edges():
        graph.edges()[u, v]['weight'] = random.random()
    return graph


def wilson(graph) -> nx.DiGraph:
    # TODO: implement
    stoch_graph = normalize_graph_weights(graph)
    random_tree = nx.DiGraph()
    return random_tree


def normalize_graph_weights(graph) -> nx.DiGraph:
    # TODO: implement
    adj_mat = nx.to_numpy_array(graph)
    norm_graph = nx.DiGraph()
    return norm_graph


if __name__ == '__main__':
    n_nodes = 10
    sample_size = 10
    log_scale_weights = False  # change
    graph = random_graph(n_nodes)

    start = time.time()
    # networkx
    trees = [nx.random_spanning_tree(graph, weight='weight',
                                     multiplicative=not log_scale_weights) for _ in range(sample_size)]
    end = time.time() - start
    print(f"K = {n_nodes}: sampled {sample_size} trees in {end}s")
    for tree in trees:
        print(f"\t{[e for e in tree.edges()]}")

